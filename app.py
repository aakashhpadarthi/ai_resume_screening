import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import tempfile  # Import tempfile to handle temporary file paths

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""  # Handle cases where extract_text returns None
        return text
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
        return ""

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Function to create a download link for the results
def create_download_link(df, filename="results.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Function to send email with attached results
def send_email_with_attachment(subject, body, to_email, attachment_path):
    from_email = os.getenv('SENDER_EMAIL')  # Use environment variable or secrets.toml
    from_password = os.getenv('SENDER_PASSWORD')  # Use environment variable or secrets.toml

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach body of the email
    msg.attach(MIMEText(body, 'plain'))

    # Attach the CSV file
    with open(attachment_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={attachment_path}")
        msg.attach(part)

    # Send the email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, from_password)
            server.sendmail(from_email, to_email, msg.as_string())
        st.success(f"Email sent to {to_email} successfully!")
    except Exception as e:
        st.error(f"Error sending email: {e}")

# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")

# Sidebar for additional options
with st.sidebar:
    st.header("Options")
    threshold = st.slider("Set a similarity score threshold", 0.0, 1.0, 0.5)
    top_n = st.number_input("Number of top resumes to display", min_value=1, value=5)

# Job description input
st.header("Job Description")
job_description_option = st.radio("Choose job description input method:", ("Text Input", "Upload PDF"))

job_description = ""  # Initialize variable to store job description

if job_description_option == "Text Input":
    job_description = st.text_area("Enter the job description")
    save_button = st.button("Save Job Description")  # Save button for text input

elif job_description_option == "Upload PDF":
    job_description_file = st.file_uploader("Upload job description as PDF", type=["pdf"])
    if job_description_file:
        job_description = extract_text_from_pdf(job_description_file)
    save_button = st.button("Save Job Description")  # Save button for PDF input

# Save button action
if save_button:
    if job_description:
        st.success("Job Description Saved!")
    else:
        st.error("Please provide a valid job description.")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")

    resumes = []  # Initialize as an empty list
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Display scores
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    # Filter results based on threshold
    filtered_results = results[results["Score"] >= threshold]

    # Display top N resumes
    st.subheader(f"Top {top_n} Resumes")
    st.write(filtered_results.head(top_n))

    # Display all results in an expander
    with st.expander("View All Results"):
        st.write(results)

    # Download link for results
    st.markdown(create_download_link(results), unsafe_allow_html=True)

    # Highlight keywords in top resumes
    st.subheader("Keyword Highlights in Top Resumes")
    vectorizer = TfidfVectorizer()
    job_description_vector = vectorizer.fit_transform([job_description])
    job_description_keywords = vectorizer.get_feature_names_out()

    for i, (resume, score) in enumerate(zip(filtered_results["Resume"].head(top_n), filtered_results["Score"].head(top_n))):
        st.write(f"*Resume:* {resume} (Score: {score:.2f})")
        resume_text = resumes[uploaded_files.index([file for file in uploaded_files if file.name == resume][0])]
        highlighted_text = resume_text
        for keyword in job_description_keywords:
            if keyword in resume_text:
                highlighted_text = highlighted_text.replace(keyword, f"<b>{keyword}</b>")
        st.markdown(highlighted_text, unsafe_allow_html=True)
        st.write("---")

    # Email sending feature
    email_input = st.text_input("Enter recipient's email address:")
    email_button = st.button("Send Results via Email")

    if email_button and email_input:
        # Create a temporary file to save the results CSV
        with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='') as tmp_file:
            results.to_csv(tmp_file, index=False)
            tmp_file_path = tmp_file.name
        
        # Send email with the results CSV file attached
        send_email_with_attachment(
            subject="AI Resume Screening Results",
            body="Please find attached the results of the AI resume screening.",
            to_email=email_input,
            attachment_path=tmp_file_path
        )

else:
    st.warning("Please upload resumes and provide a job description to proceed.")
