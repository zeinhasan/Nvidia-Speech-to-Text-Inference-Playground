import streamlit as st
from pytubefix import YouTube
from pytubefix.cli import on_progress
import torchaudio
import tempfile
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Load Whisper ASR Model (cached)
@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "openai/whisper-small"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

# Download .m4a audio only
def download_m4a_audio(youtube_url, progress_callback):
    yt = YouTube(youtube_url, on_progress_callback=on_progress)
    stream = yt.streams.filter(only_audio=True, file_extension="mp4").order_by("abr").desc().first()
    
    if not stream:
        raise RuntimeError("Audio stream tidak ditemukan.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio:
        audio_path = temp_audio.name
        stream.download(filename=audio_path)

    progress_callback(100, "✅ Audio berhasil diunduh.")
    return audio_path

# Baca waveform dari .m4a
def load_waveform(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy(), 16000

# Transkripsi
def transcribe(waveform, sample_rate, pipe, progress_callback):
    sample = {"array": waveform, "sampling_rate": sample_rate}
    progress_callback(30, "🔁 Running Whisper...")
    result = pipe(
        sample,
        return_timestamps=True,
        chunk_length_s=30,
        stride_length_s=(5, 5),
    )
    progress_callback(100, "✅ Transkripsi selesai.")
    return result["chunks"]

# UI
st.title("🎙️ YouTube Audio Transcriber (.m4a tanpa ffmpeg)")

youtube_url = st.text_input("Masukkan URL YouTube:")

if youtube_url:
    st.video(youtube_url)

    if st.button("🚀 Start Transcription"):
        # Step 1: Load model
        st.subheader("1️⃣ Load Model")
        model_progress = st.progress(0, text="Loading Whisper...")
        pipe = load_model()
        model_progress.progress(100, text="✅ Model siap.")

        # Step 2: Download audio
        st.subheader("2️⃣ Download Audio (.m4a)")
        audio_progress = st.progress(0, text="Downloading...")
        audio_path = download_m4a_audio(youtube_url, lambda val, msg: audio_progress.progress(val, text=msg))

        # Step 3: Load waveform
        st.subheader("3️⃣ Baca Audio")
        waveform, sr = load_waveform(audio_path)
        st.success("🎧 Audio berhasil dibaca.")

        # Step 4: Transcribe
        st.subheader("4️⃣ Transcribe")
        transcribe_progress = st.progress(0, text="Transcribing...")
        segments = transcribe(waveform, sr, pipe, lambda val, msg: transcribe_progress.progress(val, text=msg))

        # Step 5: Tampilkan hasil
        st.subheader("📄 Hasil Transkripsi")
        full_text = ""
        for seg in segments:
            line = f"{round(seg['timestamp'][0],2)}s - {round(seg['timestamp'][1],2)}s: {seg['text'].strip()}"
            full_text += line + "\n"
            st.markdown(f"🕒 `{round(seg['timestamp'][0],2)}s - {round(seg['timestamp'][1],2)}s` → {seg['text'].strip()}")

        st.download_button("⬇️ Download Transkripsi (.txt)", full_text, file_name="transkripsi.txt")
