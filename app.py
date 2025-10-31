# ===============================
# AI Resume Screening using NLP Embeddings (Semantic Similarity)
# ===============================

import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import base64
import tempfile
import html

# ------------------------------
# 1. PDF Text Extraction
# ------------------------------
def extract_text_from_pdf(file):
    """Extract text from uploaded PDF."""
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
        return ""

# ------------------------------
# 2. Compute Semantic Similarity
# ------------------------------
@st.cache_resource
def load_model():
    """Load the SentenceTransformer model (cached)."""
    return SentenceTransformer("all-MiniLM-L6-v2")

def rank_resumes_semantic(job_description, resumes):
    """Compute semantic similarity between job description and resumes."""
    model = load_model()
    embeddings = model.encode([job_description] + resumes, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings[0], embeddings[1:]).cpu().numpy().flatten()
    return cosine_scores

# ------------------------------
# 3. CSV Download Link
# ------------------------------
def create_download_link(df, filename="results.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}"> Download CSV File</a>'

# ------------------------------
# 4. Streamlit App
# ------------------------------
# Set page config to wide layout
st.set_page_config(page_title="AI Resume Screener (Semantic NLP)",layout="wide")

# Custom CSS for larger fonts and better line spacing
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 32px !important;       /* Bigger letters */
        line-height: 1.8 !important;      /* More spacing between lines */
    }
    .stTextInput, .stTextArea, .stNumberInput {
        font-size: 28px !important;       /* Inputs larger */
    }
    .css-1d391kg p {                     /* Paragraph spacing */
        line-height: 1.8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("Upload a job description and candidate resumes to automatically rank them using semantic similarity.")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Options")
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, step=0.05)
    top_n = st.number_input("Number of top resumes", min_value=1, value=5)
    st.markdown("---")
    st.caption("Model: `all-MiniLM-L6-v2` (Sentence Transformers)")

# Step 1: Job Description
st.header(" Step 1: Provide Job Description")
option = st.radio("Input method:", ["Type manually", "Upload PDF"])

job_description = ""
if option == "Type manually":
    job_description = st.text_area("Enter job description")
else:
    jd_file = st.file_uploader("Upload job description (PDF)", type=["pdf"])
    if jd_file:
        job_description = extract_text_from_pdf(jd_file)

# Step 2: Upload Resumes
st.header("📂 Step 2: Upload Candidate Resumes")
uploaded_files = st.file_uploader("Upload resumes (PDF)", type=["pdf"], accept_multiple_files=True)

# Step 3: Rank Resumes
if uploaded_files and job_description.strip():
    st.header("📊 Step 3: Ranking Results")

    resumes_text = []
    progress = st.progress(0)
    for i, file in enumerate(uploaded_files):
        text = extract_text_from_pdf(file)
        if len(text) < 50:
            st.warning(f"⚠️ {file.name} might be empty or image-based.")
        resumes_text.append(text)
        progress.progress((i + 1) / len(uploaded_files))

    with st.spinner("Calculating semantic similarity..."):
        scores = rank_resumes_semantic(job_description, resumes_text)

    results = pd.DataFrame({
        "Resume": [file.name for file in uploaded_files],
        "Semantic Score": scores
    }).sort_values(by="Semantic Score", ascending=False)

    # Filter based on threshold
    filtered = results[results["Semantic Score"] >= threshold]

    # Display top results
    st.subheader(f" Top {top_n} Resumes")
    st.write(filtered.head(top_n))

    with st.expander(" View All Results"):
        st.dataframe(results)

    # Download link
    st.markdown(create_download_link(results), unsafe_allow_html=True)

    # ------------------------------
    # Step 4: Keyword Highlights
    # ------------------------------
    st.subheader(" Semantic Highlights (Closest Sentences)")
    model = load_model()

    for _, row in filtered.head(top_n).iterrows():
        resume_name = row["Resume"]
        resume_text = resumes_text[uploaded_files.index(
            [f for f in uploaded_files if f.name == resume_name][0]
        )]
        
        # Split into sentences
        sentences = [s.strip() for s in resume_text.split("\n") if len(s.strip()) > 10]
        if not sentences:
            continue

        # Compute similarity with each sentence
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        jd_embedding = model.encode(job_description, convert_to_tensor=True)
        scores_sentences = util.cos_sim(jd_embedding, sentence_embeddings)[0].cpu().numpy()

        top_idx = scores_sentences.argsort()[-3:][::-1]
        top_sentences = [sentences[i] for i in top_idx]

        st.markdown(f"** {resume_name} (Score: {row['Semantic Score']:.2f})**")
        for s in top_sentences:
            st.markdown(f">  _{html.escape(s)}_")
        st.markdown("---")

else:
    st.info("Please provide a job description and upload resumes to start ranking.")
