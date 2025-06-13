from typing import List
from langchain.agents import tool
import PyPDF2
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from docx import Document
from io import BytesIO

# Return Tavily search tool instance
def tavily():
    return TavilySearchResults(max_results=5)

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Run graph and collect intermediate agent outputs
def get_agent_outputs(initial_state, graph):
    agent_outputs = []
    state = {
        "messages": [
            HumanMessage(content=initial_state["resume"], name="resume"),
            HumanMessage(content=initial_state["job_description"], name="job_description")
        ],
        "next": "supervisor"
    }

    for s in graph.stream(state):
        if "__end__" not in s:
            agent_outputs.append(s)
    return agent_outputs

# Parse final output into resume and cover letter text
def get_resume_and_coverletter_text_modified(agent_outputs):
    resume_text, cover_letter_text = [], []

    for output in agent_outputs:
        if 'Resume Editor' in output:
            resume_text.append(output['Resume Editor'])
        elif 'CoverLetter Generator' in output:
            cover_letter_text.append(output['CoverLetter Generator'])

    return resume_text[-1] if resume_text else {}, cover_letter_text[-1] if cover_letter_text else {}

# Save text to Word document
def save_text_to_docx(text, title):
    doc = Document()
    for line in text.split('\n'):
        doc.add_paragraph(line)
    return doc

# Extract plain string content from HumanMessage objects
def extract_text_from_human_message(message):
    if isinstance(message.content, str):
        return message.content
    elif isinstance(message.content, list):
        extracted_text = []
        for item in message.content:
            if isinstance(item, str):
                extracted_text.append(item)
            elif isinstance(item, dict) and 'content' in item:
                extracted_text.append(item['content'])
        return '\n'.join(extracted_text)
    return ""
