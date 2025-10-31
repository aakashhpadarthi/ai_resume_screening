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
