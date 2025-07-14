import streamlit as st
import tempfile
import torch
import nemo.collections.asr as nemo_asr

st.set_page_config(page_title="🎙️ NeMo ASR CPU", layout="centered")

@st.cache_resource
def load_model_cpu():
    # Load NeMo ASR model dan paksa ke CPU
    model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    return model.to(torch.device("cpu"))

asr_model = load_model_cpu()

st.title("🎙️ NeMo ASR Transcription (CPU Only)")
st.markdown(
    "Upload file `.wav` (mono, 16kHz) untuk ditranskripsi menggunakan model "
    "**Parakeet-TDT 0.6B** dari NVIDIA NeMo Toolkit."
)

uploaded_file = st.file_uploader("📁 Upload Audio", type=["wav"])

if uploaded_file:
    # Simpan ke file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(tmp_path)

    with st.spinner("⏳ Transcribing on CPU (this may take 1–2 minutes)..."):
        output = asr_model.transcribe([tmp_path], timestamps=True)
        segments = output[0].timestamp['segment']

    st.success("✅ Transcription Completed!")
    st.subheader("📄 Transcription with Segments & Timestamps")

    for seg in segments:
        st.write(f"🕒 `{seg['start']}s` - `{seg['end']}s`: {seg['segment']}")
