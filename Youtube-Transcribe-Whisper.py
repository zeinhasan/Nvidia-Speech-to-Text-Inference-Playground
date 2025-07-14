import streamlit as st
import tempfile, os
from pytubefix import YouTube
from pytubefix.cli import on_progress
from pydub import AudioSegment
import torchaudio
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device=="cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small",
        torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True).to(device)
    proc = AutoProcessor.from_pretrained("openai/whisper-small")
    return pipeline("automatic-speech-recognition", model=model,
        tokenizer=proc.tokenizer, feature_extractor=proc.feature_extractor,
        torch_dtype=dtype, device=device)

def convert_any_audio_to_wav(input_path):
    ext = os.path.splitext(input_path)[1][1:].lower()
    audio = AudioSegment.from_file(input_path, format=ext)
    audio = audio.set_frame_rate(16000).set_channels(1)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out:
        wav_path = out.name
        audio.export(wav_path, format="wav")
    return wav_path

def download_and_convert(youtube_url, progress):
    progress(10, "Mengunduh audio…")
    yt = YouTube(youtube_url, on_progress_callback=on_progress)
    stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
    if not stream:
        raise RuntimeError("Audio stream tidak ditemukan.")
    with tempfile.TemporaryDirectory() as d:
        out = stream.download(output_path=d)
        progress(60, f"File terunduh: {os.path.basename(out)}")
        wav = convert_any_audio_to_wav(out)
    progress(100, "Audio siap ✅")
    return wav

def transcribe(wav_path, pipe, progress):
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    sample = {"array": waveform.squeeze().numpy(), "sampling_rate": 16000}
    progress(10, "Transkripsi…")
    res = pipe(sample, return_timestamps=True, chunk_length_s=30, stride_length_s=(5,5))
    progress(100, "Selesai ✅")
    return res["chunks"]

st.title("🧠 YouTube → Audio → Whisper Transcriber")

url = st.text_input("YouTube URL")
if url and st.button("Transcribe"):
    pipe = load_model()
    st.subheader("1️⃣ Unduh & Konversi Audio")
    pb1 = st.progress(0, text="Mulai")
    wav_path = download_and_convert(url, lambda p,t: pb1.progress(p, text=t))
    st.download_button("⬇️ Unduh WAV", open(wav_path,"rb"), file_name="audio.wav")

    st.subheader("2️⃣ Transkripsi")
    pb2 = st.progress(0, text="Mulai")
    segments = transcribe(wav_path, pipe, lambda p,t: pb2.progress(p, text=t))
    
    st.success("🎉 Transkripsi Selesai")
    txt = ""
    for s in segments:
        txt += f"{s['start']}s-{s['end']}s: {s['text']}\n"
        st.markdown(f"`{s['start']}s - {s['end']}s` → {s['text']}")
    st.download_button("⬇️ Unduh Teks", txt, file_name="transkripsi.txt")
