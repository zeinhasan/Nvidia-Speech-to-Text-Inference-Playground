import streamlit as st
import tempfile
import os
from pytubefix import YouTube
from pytubefix.cli import on_progress
from pydub import AudioSegment
import torchaudio
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# ---------------------- Load Whisper Model ----------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "openai/whisper-small"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )
    return pipe

# ---------------------- Convert m4a to wav ----------------------
def convert_any_audio_to_wav(input_path):
    try:
        # Deteksi format dari ekstensi
        ext = os.path.splitext(input_path)[1][1:]  # Contoh: 'webm', 'm4a'

        audio = AudioSegment.from_file(input_path, format=ext)
        audio = audio.set_frame_rate(16000).set_channels(1)  # Mono + 16kHz

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out_file:
            wav_path = out_file.name
            audio.export(wav_path, format="wav")

        return wav_path
    except Exception as e:
        raise RuntimeError(f"Gagal mengonversi audio: {e}")

# ---------------------- Download YouTube Audio ----------------------
def download_audio_fallback(youtube_url, progress_callback):
    progress_callback(10, "Mengunduh audio dari YouTube...")
    yt = YouTube(youtube_url, on_progress_callback=on_progress)

    stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()

    if stream is None:
        raise ValueError("Tidak ditemukan stream audio.")

    ext = stream.subtype or "audio"
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as temp_audio:
        audio_path = temp_audio.name
        stream.download(filename=audio_path)

    progress_callback(60, f"Mengonversi ke WAV dari .{ext}...")
    wav_path = convert_any_audio_to_wav(audio_path)

    progress_callback(100, "Audio siap ✅")
    return wav_path


# ---------------------- Transkripsi ----------------------
def transcribe_audio(wav_path, pipe, progress_callback):
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    sample = {"array": waveform.squeeze().numpy(), "sampling_rate": 16000}

    progress_callback(10, "Menjalankan transkripsi...")
    result = pipe(sample, return_timestamps=True, chunk_length_s=30, stride_length_s=(5, 5))

    segments = result["chunks"]
    progress_callback(100, "Transkripsi selesai ✅")
    return [
        {
            "start": round(seg["timestamp"][0], 2),
            "end": round(seg["timestamp"][1], 2),
            "text": seg["text"].strip()
        }
        for seg in segments
    ]

# ---------------------- Streamlit UI ----------------------
st.title("🎙️ YouTube Audio Transcriber")

youtube_url = st.text_input("Masukkan link YouTube:")

if youtube_url:
    st.video(youtube_url)

    if st.button("🚀 Mulai Transkripsi"):
        st.subheader("1️⃣ Memuat model...")
        model_progress = st.progress(0, text="Inisialisasi model...")
        pipe = load_model()
        model_progress.progress(100, text="Model siap ✅")

        st.subheader("2️⃣ Mengunduh audio...")
        audio_progress = st.progress(0)
        wav_path = download_audio_m4a(youtube_url, lambda v, t: audio_progress.progress(v, text=t))

        st.success("✅ Audio berhasil diunduh dan dikonversi!")
        with open(wav_path, "rb") as f:
            st.download_button("⬇️ Unduh Audio WAV", f, file_name="audio.wav")

        st.subheader("3️⃣ Menjalankan transkripsi...")
        transcribe_progress = st.progress(0)
        segments = transcribe_audio(wav_path, pipe, lambda v, t: transcribe_progress.progress(v, text=t))

        st.subheader("📄 Hasil Transkripsi")
        full_text = ""
        for seg in segments:
            line = f"{seg['start']}s - {seg['end']}s: {seg['text']}"
            full_text += line + "\n"
            st.markdown(f"🕒 `{seg['start']}s - {seg['end']}s` → {seg['text']}")

        st.download_button("⬇️ Unduh Transkripsi (.txt)", full_text, file_name="transkripsi.txt")
