# 📄 Resume Analyzer

Streamlit app that parses PDF resumes into structured data and matches them
against a job description — powered by **LangChain + Groq**.

## Features

- Upload a PDF resume → LLM extracts name, email, phone, skills, education, experience
- Paste a job description → LLM scores the match, lists matched/missing skills,
  strengths, gaps, and a summary

## Project structure

```
resume-analyzer/
├── app.py              # Streamlit UI
├── parser.py           # PDF text extraction + resume -> structured data (Groq LLM)
├── matcher.py           # Resume vs JD matching (Groq LLM)
├── requirements.txt
├── .env.example
└── uploads/             # uploaded resumes are saved here
```

## Setup

1. **Get a Groq API key** — free at https://console.groq.com/keys

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API key**

   Copy `.env.example` to `.env` and fill in your key:

   ```bash
   cp .env.example .env
   ```

   ```
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

   It will open at `http://localhost:8501`.

## Usage

1. Upload a PDF resume in the sidebar and click **Parse Resume**.
2. Review the extracted candidate info, skills, education, and experience
   (raw JSON is available in the expander).
3. Paste a job description in the **Match Against a Job Description** section
   and click **Run JD Match** to get a score, matched/missing skills,
   strengths, gaps, and a summary.

## Notes

- The default model is `llama-3.3-70b-versatile` (set in `parser.py` /
  `get_llm()`). Swap it for any other Groq-hosted model if you like.
- Only text-based PDFs are supported out of the box (scanned/image-only PDFs
  need OCR first).
