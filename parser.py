# ==========================================
# parser.py
# PDF text extraction + LLM-based resume parsing (Groq via LangChain)
# ==========================================

import io
from typing import List, Optional

from pydantic import BaseModel, Field
from pypdf import PdfReader

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# ==========================================
# Data models
# ==========================================

class Education(BaseModel):
    degree: str = Field(description="Degree name, e.g. 'B.Tech in Computer Science'")
    university_name: str = Field(description="Name of the university / college / institute")
    gpa: Optional[float] = Field(default=None, description="GPA or CGPA if mentioned in the resume")


class Experience(BaseModel):
    company_name: Optional[str] = Field(default=None, description="Company or organization name")
    years: Optional[float] = Field(default=None, description="Number of years worked at this role/company")
    project_name: Optional[str] = Field(default=None, description="Name of a key project, if mentioned")
    tech_stack: Optional[str] = Field(default=None, description="Technologies / tools used, comma separated")
    project_description: Optional[str] = Field(default=None, description="Short description of the work or project")


class Resume(BaseModel):
    name: Optional[str] = Field(default=None, description="Candidate's full name")
    email: Optional[str] = Field(default=None, description="Candidate's email address")
    phone_number: Optional[str] = Field(default=None, description="Candidate's phone number")
    skills: List[str] = Field(default_factory=list, description="List of technical and soft skills")
    education: List[Education] = Field(default_factory=list, description="Education history, most recent first")
    experience: List[Experience] = Field(default_factory=list, description="Work experience / projects, most recent first")


# ==========================================
# LLM setup
# ==========================================

def get_llm(model: str = "llama-3.3-70b-versatile", temperature: float = 0.0) -> ChatGroq:
    """
    Return a configured Groq chat model.
    Reads GROQ_API_KEY from the environment (set via .env / load_dotenv() in app.py).
    """
    return ChatGroq(model=model, temperature=temperature)


# ==========================================
# PDF text extraction
# ==========================================

def load_resume_text_from_bytes(file_bytes: bytes) -> str:
    """Extract raw text from a PDF resume given as raw bytes."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages_text).strip()

    if not text:
        raise ValueError(
            "Could not extract any text from this PDF. "
            "It may be a scanned/image-only resume — please upload a text-based PDF."
        )
    return text


# ==========================================
# Resume parsing (LLM)
# ==========================================

def parse_resume(resume_text: str, llm: ChatGroq) -> Resume:
    """Use the LLM to convert raw resume text into a structured Resume object."""
    parser = PydanticOutputParser(pydantic_object=Resume)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert resume parser. Carefully read the resume text and "
                "extract structured information from it. Rules:\n"
                "- If a field is not present in the resume, leave it null/empty — never invent data.\n"
                "- Normalize skills into a clean, deduplicated list.\n"
                "- For experience entries, estimate `years` from date ranges when possible.\n\n"
                "{format_instructions}",
            ),
            ("human", "Resume text:\n\n{resume_text}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    return chain.invoke({"resume_text": resume_text})
