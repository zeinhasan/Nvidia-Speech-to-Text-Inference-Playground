import streamlit as st
import tempfile
import torch
import librosa
from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy.editor import AudioFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# 1. Load Whisper Model dan buat pipeline
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

def load_asr_model_with_progress(progress_callback):
    progress_callback(10, "📦 Loading Whisper model...")
    pipe = load_model_cached()
    progress_callback(100, "✅ Model loaded")
    return pipe

# 2. Konversi MP4 ke WAV dengan moviepy
def convert_to_wav_moviepy(video_path, audio_path):
    audio_clip = AudioFileClip(video_path)
    audio_clip = audio_clip.set_fps(16000).set_channels(1)
    audio_clip.write_audiofile(audio_path, fps=16000)

# 3. Download YouTube audio & convert ke WAV
def download_and_extract_audio(youtube_url, progress_callback):
    try:
        progress_callback(10, "🔗 Downloading YouTube audio...")
        yt = YouTube(youtube_url, on_progress_callback=on_progress)
        stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()

        if stream is None:
            raise ValueError("No audio stream found.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
            video_path = temp_vid.name
            stream.download(filename=video_path)

        progress_callback(60, "🎛️ Converting to WAV...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio_path = temp_audio.name
            convert_to_wav_moviepy(video_path, audio_path)

        progress_callback(100, "🎧 Audio siap!")
        return audio_path

    except Exception as e:
        st.error(f"❌ Gagal mengunduh/konversi: {e}")
        raise

# 4. Transkripsi audio WAV
def transcribe_audio(audio_path, pipe, progress_callback):
    waveform, sr = librosa.load(audio_path, sr=16000)
    sample = {"array": waveform, "sampling_rate": sr}

    progress_callback(10, "🧠 Running transcription...")
    result = pipe(
        sample,
        return_timestamps=True,
        chunk_length_s=30,
        stride_length_s=(5, 5),
    )

    progress_callback(100, "✅ Transkripsi selesai")
    return [
        {
            "start": round(seg["timestamp"][0], 2),
            "end": round(seg["timestamp"][1], 2),
            "text": seg["text"].strip()
        }
        for seg in result["chunks"]
    ]

# 5. Streamlit UI
st.set_page_config(page_title="YouTube ASR Transcriber", layout="centered")
st.title("🎙️ YouTube Audio Transcriber with Timestamps")

youtube_url = st.text_input("🔗 Masukkan Link YouTube:")

if youtube_url:
    st.video(youtube_url)

    if st.button("🚀 Mulai Transkripsi"):
        # Step 1: Load model
        st.subheader("1️⃣ Load Model")
        model_prog = st.progress(0, text="⏳ Memuat model...")
        pipe = load_asr_model_with_progress(lambda v, t: model_prog.progress(v, text=t))

        # Step 2: Download dan konversi
        st.subheader("2️⃣ Download & Convert Audio")
        audio_prog = st.progress(0, text="⏳ Mengunduh video...")
        audio_path = download_and_extract_audio(youtube_url, lambda v, t: audio_prog.progress(v, text=t))

        st.success("✅ Audio WAV berhasil dibuat!")
        with open(audio_path, "rb") as f:
            st.download_button("⬇️ Download Audio WAV", f, file_name="audio.wav")

        # Step 3: Transkripsi
        st.subheader("3️⃣ Transkripsi")
        trans_prog = st.progress(0, text="🧠 Menjalankan transkripsi...")
        segments = transcribe_audio(audio_path, pipe, lambda v, t: trans_prog.progress(v, text=t))

        # Step 4: Tampilkan hasil
        st.success("📄 Transkripsi selesai!")
        st.subheader("📄 Hasil Transkripsi dengan Timestamp")

        full_text = ""
        for seg in segments:
            line = f"{seg['start']}s - {seg['end']}s: {seg['text']}"
            st.markdown(f"🕒 `{seg['start']}s - {seg['end']}s` → {seg['text']}")
            full_text += line + "\n"

        st.download_button("⬇️ Download Transkripsi (.txt)", full_text, file_name="transkripsi.txt")
