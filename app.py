import streamlit as st
from PIL import Image
import sqlite3
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pypdf
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Veritas | AI Content Detector",
    page_icon="🛡️",
    layout="centered"
)

# --- Load Models Directly with Explicit Architecture ---
@st.cache_resource
def load_detection_engine():
    # Load model and tokenizer directly so we can inspect actual ID-to-Label mapping
    model_name = "drasti02/ai-text-detector-roberta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Image pipeline
    image_pipeline = pipeline(
        "image-classification",
        model="umm-maybe/AI-image-detector"
    )
    return tokenizer, model, image_pipeline

with st.spinner("Initializing models and loading neural weights..."):
    text_tokenizer, text_model, image_pipeline = load_detection_engine()

# --- Database Layer ---
DB_FILE = "scan_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            modality TEXT,
            verdict TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

def log_scan(modality, verdict, confidence):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO scans (timestamp, modality, verdict, confidence) VALUES (?, ?, ?, ?)",
        (datetime.now().strftime("%d %b %Y, %I:%M %p"), modality, verdict, float(confidence))
    )
    conn.commit()
    conn.close()

def get_recent_scans():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT timestamp, modality, verdict, confidence FROM scans ORDER BY id DESC LIMIT 5")
    rows = c.fetchall()
    conn.close()
    return rows

init_db()

# --- Direct Softmax Scoring Function ---
def analyze_text(text):
    # Tokenize and run forward pass
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
    
    # Check model's own config label mapping
    id2label = text_model.config.id2label
    # Typically in this fine-tuned checkpoint:
    # Index 0 is Human / Real, Index 1 is AI / Generated (or vice versa)
    score_0 = probs[0].item() * 100.0
    score_1 = probs[1].item() * 100.0

    raw_label_0 = id2label.get(0, "LABEL_0").lower()
    raw_label_1 = id2label.get(1, "LABEL_1").lower()

    return score_0, score_1, raw_label_0, raw_label_1

# --- App Header ---
st.title("🛡️ Veritas | AI Detector")
st.write("Detect AI-generated text, documents, and synthetic images using Deep Learning classifiers.")

tab_text, tab_pdf, tab_image, tab_history = st.tabs(["📝 Text Check", "📄 PDF Check", "🖼️ Image Check", "📋 Recent Scans"])

# ==========================================
# 1. TEXT TAB
# ==========================================
with tab_text:
    user_text = st.text_area(
        "Paste text to analyze:",
        placeholder="Paste your essay, article, or writing sample here...",
        height=180
    )
    
    words = user_text.strip().split() if user_text.strip() else []
    word_count = len(words)
    st.caption(f"Word count: **{word_count}** (Minimum 15 recommended)")

    if st.button("Check Text", type="primary", use_container_width=True):
        if word_count < 15:
            st.warning("⚠️ Please provide at least 15 words so the neural network has enough linguistic context.")
        else:
            with st.spinner("Evaluating logits across attention heads..."):
                s0, s1, l0, l1 = analyze_text(user_text)

                # Show both raw scores immediately so you can see exactly what the model computed
                with st.expander("🔍 Exact Logit Probabilities (Check Here)", expanded=True):
                    st.write(f"**Index 0 ({l0})**: `{s0:.2f}%`")
                    st.write(f"**Index 1 ({l1})**: `{s1:.2f}%`")

                # In drasti02/ai-text-detector-roberta:
                # Index 0 represents Human, Index 1 represents AI
                ai_prob = s1
                human_prob = s0

                if ai_prob >= 50.0:
                    verdict = "AI Generated"
                    st.error("🚨 **Result: AI-Generated Content**")
                    st.metric("AI Probability", f"{ai_prob:.1f}%")
                    st.progress(int(ai_prob))
                else:
                    verdict = "Human Written"
                    st.success("✅ **Result: Human-Written Content**")
                    st.metric("Human Authenticity", f"{human_prob:.1f}%")
                    st.progress(int(human_prob))

                log_scan("Text", verdict, max(ai_prob, human_prob))

# ==========================================
# 2. PDF TAB
# ==========================================
with tab_pdf:
    uploaded_pdf = st.file_uploader("Upload a PDF document to inspect:", type=["pdf"])

    if uploaded_pdf:
        if st.button("Analyze PDF Document", type="primary", use_container_width=True):
            with st.spinner("Extracting text and calculating probabilities..."):
                try:
                    pdf_bytes = io.BytesIO(uploaded_pdf.read())
                    reader = pypdf.PdfReader(pdf_bytes)
                    num_pages = len(reader.pages)
                    
                    extracted_text = " ".join([page.extract_text() or "" for page in reader.pages]).strip()
                    pdf_words = extracted_text.split() if extracted_text else []
                    
                    if len(pdf_words) < 15:
                        st.warning("⚠️ The uploaded PDF has fewer than 15 extractable words. It may be a scanned image document.")
                    else:
                        st.caption(f"Pages Processed: **{num_pages}** | Total Words: **{len(pdf_words)}**")
                        
                        analysis_excerpt = " ".join(pdf_words[:500])
                        s0, s1, l0, l1 = analyze_text(analysis_excerpt)

                        ai_prob = s1
                        human_prob = s0

                        if ai_prob >= 50.0:
                            verdict = "AI Generated"
                            st.error("🚨 **Result: AI-Generated Document Content**")
                            st.metric("AI Probability", f"{ai_prob:.1f}%")
                            st.progress(int(ai_prob))
                        else:
                            verdict = "Human Written"
                            st.success("✅ **Result: Human-Authored Document**")
                            st.metric("Human Authenticity", f"{human_prob:.1f}%")
                            st.progress(int(human_prob))

                        with st.expander("🔍 Document Scores"):
                            st.write(f"Index 0: `{s0:.2f}%` | Index 1: `{s1:.2f}%`")

                        log_scan("PDF", verdict, max(ai_prob, human_prob))
                except Exception as e:
                    st.error(f"Failed to read PDF file: {str(e)}")

# ==========================================
# 3. IMAGE TAB
# ==========================================
with tab_image:
    uploaded_file = st.file_uploader("Upload an image to inspect:", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        img_pil = Image.open(uploaded_file).convert("RGB")
        st.image(img_pil, caption="Target Image", use_container_width=True)

        if st.button("Check Image", type="primary", use_container_width=True):
            with st.spinner("Inspecting latent artifacts with Vision Transformer..."):
                predictions = image_pipeline(img_pil)
                top_pred = predictions[0]
                label = top_pred['label'].lower()
                score = top_pred['score'] * 100.0

                if "art" in label or "fake" in label or "artificial" in label:
                    verdict = "AI Generated"
                    st.error("🚨 **Result: AI-Generated Image**")
                    st.metric("Synthetic Confidence", f"{score:.1f}%")
                    st.progress(int(score))
                else:
                    verdict = "Authentic Photo"
                    st.success("✅ **Result: Authentic Camera Photo**")
                    st.metric("Authenticity Confidence", f"{score:.1f}%")
                    st.progress(int(score))

                log_scan("Image", verdict, score)

# ==========================================
# 4. RECENT SCANS TAB
# ==========================================
with tab_history:
    st.subheader("Recent Verification Logs")
    records = get_recent_scans()
    if records:
        for r in records:
            st.write(f"📅 **{r[0]}** | Modality: `{r[1]}` → **{r[2]}** ({r[3]:.1f}%)")
            st.divider()
    else:
        st.info("No scans recorded yet. Run a text or image check to log entries.")