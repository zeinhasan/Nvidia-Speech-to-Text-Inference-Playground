import streamlit as st
import os
from pytube import YouTube
from moviepy.editor import VideoFileClip
import tempfile
import librosa
import torch
import threading
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Load model sekali saja
@st.cache_resource
def load_asr_model():
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

pipe = load_asr_model()

# Fungsi download dan ekstrak audio
def download_and_extract_audio(youtube_url):
    yt = YouTube(youtube_url)
    video_stream = yt.streams.filter(only_audio=True).first()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        video_path = temp_video.name
        video_stream.download(filename=video_path)

    # Konversi ke WAV dengan moviepy
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_path = temp_audio.name
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000, verbose=False, logger=None)
    
    return audio_path

# Fungsi transkripsi
def transcribe_audio(audio_path):
    waveform, sample_rate = librosa.load(audio_path, sr=16000)
    sample = {"array": waveform, "sampling_rate": sample_rate}

    result = pipe(
        sample,
        return_timestamps=True,
        chunk_length_s=30,
        stride_length_s=(5, 5),
    )

    segments = result["chunks"]
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

    if st.button("🚀 Transcribe Now"):
        with st.spinner("⏳ Downloading and transcribing audio..."):

            transcribed_segments = []

            def async_transcribe():
                try:
                    audio_path = download_and_extract_audio(youtube_url)
                    segments = transcribe_audio(audio_path)
                    transcribed_segments.extend(segments)
                except Exception as e:
                    st.error(f"❌ Gagal: {e}")

            thread = threading.Thread(target=async_transcribe)
            thread.start()

            while thread.is_alive():
                st.info("🔁 Transcribing in progress...")
                st.sleep(1)
            thread.join()

        st.success("✅ Transkripsi selesai!")

        for seg in transcribed_segments:
            st.markdown(f"🕒 `{seg['start']}s - {seg['end']}s` → {seg['text']}")

