import streamlit as st
import os
from pytube import YouTube
import ffmpeg
import tempfile
import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Load model sekali saja
# ✅ Cache hanya fungsi ini
@st.cache_resource(show_spinner=False)
def load_model_cached():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "openai/whisper-large-v3-turbo"

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

# ✅ Fungsi biasa (tidak di-cache) untuk tampilkan progress
def load_asr_model_with_progress(progress_callback):
    progress_callback(0, "Loading model... (0%)")
    progress_callback(20, "Downloading model files...")
    pipe = load_model_cached()
    progress_callback(100, "Model loaded ✅")
    return pipe



# Konversi MP4 ke WAV menggunakan ffmpeg-python
def convert_to_wav_ffmpeg(video_path, audio_path):
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, ac=1, ar='16000')
        .run(quiet=True, overwrite_output=True)
    )

# Fungsi download dan ekstrak audio
def download_and_extract_audio(youtube_url, progress_callback):
    progress_callback(10, "Downloading YouTube audio...")
    yt = YouTube(youtube_url)
    video_stream = yt.streams.filter(only_audio=True).first()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        video_path = temp_video.name
        video_stream.download(filename=video_path)

    progress_callback(60, "Converting to WAV...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_path = temp_audio.name
        convert_to_wav_ffmpeg(video_path, audio_path)

    progress_callback(100, "Audio ready ✅")
    return audio_path

# Fungsi transkripsi
def transcribe_audio(audio_path, pipe, progress_callback):
    waveform, sample_rate = librosa.load(audio_path, sr=16000)
    sample = {"array": waveform, "sampling_rate": sample_rate}

    progress_callback(10, "Running transcription...")
    result = pipe(
        sample,
        return_timestamps=True,
        chunk_length_s=30,
        stride_length_s=(5, 5),
    )

    segments = result["chunks"]
    progress_callback(100, "Transcription completed ✅")
    return [
        {
            "start": round(seg["timestamp"][0], 2),
            "end": round(seg["timestamp"][1], 2),
            "text": seg["text"].strip()
        }
        for seg in segments
    ]

# Streamlit UI
st.title("🎙️ YouTube Audio Transcriber with Timestamp")

youtube_url = st.text_input("Masukkan Link YouTube:", "")

if youtube_url:
    st.video(youtube_url)

    if st.button("🚀 Start Transcription"):
        st.subheader("1️⃣ Load Model")
        model_progress = st.progress(0, text="Initializing model...")
        pipe = load_asr_model_with_progress(lambda val, msg: model_progress.progress(val, text=msg))

        st.subheader("2️⃣ Download & Convert Audio")
        audio_progress = st.progress(0, text="Downloading...")
        audio_path = download_and_extract_audio(youtube_url, lambda val, msg: audio_progress.progress(val, text=msg))

        st.success("🎧 Audio siap!")
        with open(audio_path, "rb") as f:
            st.download_button("⬇️ Download Audio WAV", f, file_name="audio.wav")

        st.subheader("3️⃣ Transcribe Audio")
        transcribe_progress = st.progress(0, text="Memulai transkripsi...")
        segments = transcribe_audio(audio_path, pipe, lambda val, msg: transcribe_progress.progress(val, text=msg))

        st.success("✅ Transkripsi selesai!")

        st.subheader("📄 Hasil Transkripsi")
        full_text = ""
        for seg in segments:
            line = f"{seg['start']}s - {seg['end']}s: {seg['text']}"
            full_text += line + "\n"
            st.markdown(f"🕒 `{seg['start']}s - {seg['end']}s` → {seg['text']}")

        # Tombol download hasil transkripsi
        st.download_button("⬇️ Download Transkripsi (.txt)", full_text, file_name="transkripsi.txt")
