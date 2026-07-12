# ==========================================
# app.py
# Streamlit UI - Resume Parser + JD Matcher
# ==========================================

import os
import json

import streamlit as st
from dotenv import load_dotenv

from parser import load_resume_text_from_bytes, parse_resume, get_llm
from matcher import match_resume_to_jd

# ==========================================
# Setup
# ==========================================

load_dotenv()  # reads GROQ_API_KEY from .env

st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")
st.title("📄 Resume Analyzer")
st.caption("Parse resumes and match them against a job description using LangChain + Groq.")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY is not set. Create a `.env` file in this folder "
        "(see `.env.example`) with your Groq API key, then restart the app."
    )
    st.stop()

# ==========================================
# Session state
# ==========================================

if "resume" not in st.session_state:
    st.session_state.resume = None
if "match_result" not in st.session_state:
    st.session_state.match_result = None

# ==========================================
# Sidebar - Resume Upload
# ==========================================

with st.sidebar:
    st.header("1. Upload Resume")
    uploaded_file = st.file_uploader("Upload a PDF resume", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Parse Resume", type="primary"):
            with st.spinner("Reading and parsing resume..."):
                # Save a copy to uploads/ for reference
                save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                resume_text = load_resume_text_from_bytes(uploaded_file.getvalue())
                llm = get_llm()
                resume = parse_resume(resume_text, llm=llm)
                st.session_state.resume = resume
                st.session_state.match_result = None  # reset previous match
            st.success("Resume parsed successfully!")

# ==========================================
# Main - Parsed Resume Display
# ==========================================

if st.session_state.resume:
    resume = st.session_state.resume

    st.header("Parsed Resume")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Candidate Info")
        st.write(f"**Name:** {resume.name}")
        st.write(f"**Email:** {resume.email}")
        st.write(f"**Phone:** {resume.phone_number}")

        if resume.skills:
            st.subheader("Skills")
            st.write(", ".join(resume.skills))

    with col2:
        if resume.education:
            st.subheader("Education")
            for edu in resume.education:
                gpa_str = f" — GPA: {edu.gpa}" if edu.gpa is not None else ""
                st.write(f"- **{edu.degree}**, {edu.university_name}{gpa_str}")

        if resume.experience:
            st.subheader("Experience")
            for exp in resume.experience:
                st.write(f"- **{exp.company_name or 'N/A'}** ({exp.years or '?'} yrs)")
                if exp.project_name:
                    st.write(f"  Project: {exp.project_name}")
                if exp.tech_stack:
                    st.write(f"  Tech Stack: {exp.tech_stack}")
                if exp.project_description:
                    st.write(f"  {exp.project_description}")

    with st.expander("Raw JSON"):
        st.json(resume.model_dump())

    st.divider()

    # ==========================================
    # JD Matching Section
    # ==========================================

    st.header("2. Match Against a Job Description")
    jd_text = st.text_area("Paste the job description here", height=200)

    if st.button("Run JD Match", disabled=not jd_text.strip()):
        with st.spinner("Comparing resume to job description..."):
            llm = get_llm()
            match_result = match_resume_to_jd(resume, jd_text, llm=llm)
            st.session_state.match_result = match_result

    if st.session_state.match_result:
        match_result = st.session_state.match_result

        st.subheader("Match Results")
        st.metric("Match Score", f"{match_result.match_score:.0f}/100")

        col3, col4 = st.columns(2)
        with col3:
            st.write("**✅ Matched Skills**")
            st.write(", ".join(match_result.matched_skills) or "None found")

            st.write("**💪 Strengths**")
            for s in match_result.strengths:
                st.write(f"- {s}")

        with col4:
            st.write("**❌ Missing Skills**")
            st.write(", ".join(match_result.missing_skills) or "None")

            st.write("**⚠️ Gaps**")
            for g in match_result.gaps:
                st.write(f"- {g}")

        st.write("**Summary**")
        st.write(match_result.summary)

else:
    st.info("Upload a resume PDF from the sidebar and click **Parse Resume** to get started.")
