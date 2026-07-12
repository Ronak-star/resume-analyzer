# ==========================================
# matcher.py
# LLM-based resume <-> job description matching (Groq via LangChain)
# ==========================================

from typing import List

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from parser import Resume


# ==========================================
# Data model
# ==========================================

class MatchResult(BaseModel):
    match_score: float = Field(description="Overall match score out of 100")
    matched_skills: List[str] = Field(default_factory=list, description="Skills required by the JD that the candidate has")
    missing_skills: List[str] = Field(default_factory=list, description="Skills required by the JD that the candidate is missing")
    strengths: List[str] = Field(default_factory=list, description="Key strengths of the candidate relevant to this role")
    gaps: List[str] = Field(default_factory=list, description="Key gaps or weaknesses relevant to this role")
    summary: str = Field(description="A concise 2-4 sentence summary of the overall fit for the role")


# ==========================================
# Matching (LLM)
# ==========================================

def match_resume_to_jd(resume: Resume, jd_text: str, llm: ChatGroq) -> MatchResult:
    """Compare a parsed Resume against a job description and return a structured match report."""
    parser = PydanticOutputParser(pydantic_object=MatchResult)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert technical recruiter. Compare the candidate's resume "
                "(given below as JSON) against the job description and produce an honest, "
                "well-reasoned match assessment. Base match_score on how well the candidate's "
                "skills and experience align with the JD's requirements. Be specific and concrete "
                "in strengths, gaps, and the summary — avoid generic filler.\n\n"
                "{format_instructions}",
            ),
            (
                "human",
                "Candidate resume (JSON):\n{resume_json}\n\nJob description:\n{jd_text}",
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    return chain.invoke(
        {
            "resume_json": resume.model_dump_json(indent=2),
            "jd_text": jd_text,
        }
    )
