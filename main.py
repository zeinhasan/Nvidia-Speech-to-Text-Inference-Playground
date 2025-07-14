import streamlit as st
from nemo.collections.asr.models import ASRModel
import tempfile
import os

# Load model once at start
@st.cache_resource
def load_model():
    return ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

asr_model = load_model()

st.title("🎙️ NeMo ASR Transcriber with Timestamp")
st.write("Upload file `.wav` untuk ditranskrip otomatis.")

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    st.audio(tmp_file_path)

    with st.spinner("⏳ Transcribing..."):
        output = asr_model.transcribe([tmp_file_path], timestamps=True)
        word_timestamps = output[0].timestamp['word']
        segment_timestamps = output[0].timestamp['segment']
    
    st.success("✅ Transcription completed!")

    st.subheader("📄 Transcription with Timestamps")
    for stamp in segment_timestamps:
        st.write(f"🕒 {stamp['start']}s - {stamp['end']}s: {stamp['segment']}")

    os.remove(tmp_file_path)
