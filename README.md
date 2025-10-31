# 🤖 AI Resume Screener (Semantic NLP)

Automatically rank resumes based on their **semantic similarity** to a given **job description** using **Natural Language Processing (NLP)** and **Sentence Transformers**.

---

## 🚀 Overview

This Streamlit web app uses **NLP embeddings** to evaluate how closely each resume aligns with a job description — not just by keyword matching, but by **semantic meaning**.

Upload a job description and multiple candidate resumes (PDFs), and the app will:
- 🧠 Extract text from each PDF
- 🔍 Compute **semantic similarity** using a transformer model (`all-MiniLM-L6-v2`)
- 📊 Rank resumes based on similarity scores
- 💾 Provide an option to **download results** as a CSV file
- 💬 Highlight **key matching sentences** for deeper insight

---

## 🧩 Features

✅ Extracts text from PDFs using `PyPDF2`  
✅ Computes **semantic similarity** using `sentence-transformers`  
✅ Displays top-matching resumes interactively  
✅ Downloadable CSV report  
✅ Clean, large-font UI built with Streamlit  
✅ Cached model loading for fast performance  

---

## 🛠️ Tech Stack

| Component | Library / Framework |
|------------|---------------------|
| Frontend | [Streamlit](https://streamlit.io/) |
| NLP Model | [Sentence Transformers](https://www.sbert.net/) |
| Embeddings | `all-MiniLM-L6-v2` |
| PDF Parsing | `PyPDF2` |
| Data Handling | `pandas` |
| File Encoding | `base64` |

---

## ⚙️ Installation & Setup

### 1. Clone this repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

###📂 Usage

Step 1: Enter or upload a job description (PDF)

Step 2: Upload one or more candidate resumes (PDF)

Step 3: Click to rank resumes

View top results, download a CSV report, and review semantic highlights
The higher the score, the better the alignment between the resume and the job description.

###💡 How It Works

The job description and each resume are encoded into high-dimensional vectors using a pretrained transformer (all-MiniLM-L6-v2).

The app calculates the cosine similarity between the job description vector and each resume vector.

The results are sorted, filtered by a similarity threshold, and displayed interactively.

###🧰 Customization

You can adjust:

Similarity Threshold: Filter out low-similarity resumes

Top N Results: Control how many best resumes to display

Model Name: Try other SentenceTransformer models for improved accuracy

###📦 Download Results

Click “Download CSV File” to export the ranked list of resumes along with their similarity scores.

### Future Enhancements

🗂️ Support for DOCX and text files

🔍 Keyword extraction and visualization

📈 Enhanced resume analytics (skills heatmap)

⚡ GPU acceleration with Torch CUDA

###🧑‍💻 Author

Aakash Padarthi
🎓 Student & AI Enthusiast

