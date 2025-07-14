import streamlit as st
from pytubefix import YouTube
from pytubefix.cli import on_progress
import tempfile
import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# 🔁 Load Whisper Model
@st.cache_resource(show_spinner=False)
def load_model_cached():
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

# 🎧 Download audio from YouTube and convert to waveform
def download_youtube_audio_as_waveform(url, progress_callback):
    try:
        progress_callback(10, "Mengunduh audio dari YouTube...")
        yt = YouTube(url, on_progress_callback=on_progress)
        stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()

        if stream is None:
            raise ValueError("Tidak menemukan audio stream.")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_audio_file:
            stream.download(filename=temp_audio_file.name)
            video_path = temp_audio_file.name

        progress_callback(60, "Membaca dan memproses audio...")
        waveform, _ = librosa.load(video_path, sr=16000)
        progress_callback(100, "Audio siap!")
        return waveform, video_path

    except Exception as e:
        st.error(f"❌ Gagal mengambil audio: {e}")
        raise

# ✍️ Transcribe
def transcribe_audio_from_waveform(waveform, pipe, progress_callback):
    progress_callback(10, "Menjalankan transkripsi...")
    sample = {"array": waveform, "sampling_rate": 16000}
    result = pipe(
        sample,
        return_timestamps=True,
        chunk_length_s=30,
        stride_length_s=(5, 5),
    )
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

# 🎛️ Streamlit UI
st.set_page_config(page_title="YouTube Whisper Transcriber", layout="centered")
st.title("🎙️ YouTube Audio Transcriber with Timestamp")

youtube_url = st.text_input("🔗 Masukkan link YouTube:", "")

if youtube_url:
    st.video(youtube_url)

    if st.button("🚀 Transcribe Sekarang"):
        # 1. Load Model
        st.subheader("1️⃣ Load Model")
        model_progress = st.progress(0, text="Menyiapkan model...")
        pipe = load_model_cached()
        model_progress.progress(100, text="Model siap ✅")

        # 2. Download Audio
        st.subheader("2️⃣ Download Audio")
        audio_progress = st.progress(0, text="Mengunduh...")
        waveform, raw_audio_path = download_youtube_audio_as_waveform(
            youtube_url, lambda v, t: audio_progress.progress(v, text=t)
        )

        # Tombol download audio mentah
        with open(raw_audio_path, "rb") as f:
            st.download_button("⬇️ Download File Audio Mentah (.mp4)", f, file_name="audio.mp4")

        # 3. Transkripsi
        st.subheader("3️⃣ Transkripsi")
        transcribe_progress = st.progress(0, text="Transcribing...")
        segments = transcribe_audio_from_waveform(waveform, pipe, lambda v, t: transcribe_progress.progress(v, text=t))

        # 4. Tampilkan Hasil
        st.success("✅ Transkripsi selesai!")
        st.subheader("📄 Hasil Transkripsi")
        full_text = ""
        for seg in segments:
            line = f"{seg['start']}s - {seg['end']}s: {seg['text']}"
            st.markdown(f"🕒 `{seg['start']}s - {seg['end']}s` → {seg['text']}")
            full_text += line + "\n"

        # 5. Download Transkripsi
        st.download_button("⬇️ Download Transkripsi (.txt)", full_text, file_name="transkripsi.txt")
