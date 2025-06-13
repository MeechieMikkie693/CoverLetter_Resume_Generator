import streamlit as st
from io import BytesIO
import os

from dotenv import load_dotenv
load_dotenv()

from agents import define_graph
from llms import load_llm
from tools_1 import (
    extract_text_from_pdf,
    get_agent_outputs,
    get_resume_and_coverletter_text_modified,
    save_text_to_docx,
    extract_text_from_human_message
)

from langchain_core.messages import HumanMessage

# Streamlit UI settings
st.set_page_config(layout="wide")
st.title(" Job_Fit_AI – Resume & Cover Letter Generator")

# File upload
uploaded_file = st.sidebar.file_uploader(" Upload Your Resume (PDF)", type="pdf")

# Job Description input
job_description = st.text_area(" Enter the Job Description")

# Ensure user uploaded file and description
if uploaded_file is not None and job_description:
    if st.button(" Generate Tailored Documents"):
        with st.spinner("Analyzing and generating..."):
            # Load Gemini model
            llm = load_llm()

            # Build agent graph
            graph = define_graph(llm)

            # Extract text from uploaded resume PDF
            resume_text = extract_text_from_pdf(uploaded_file)

            # Run agent graph
            initial_state = {
                "resume": resume_text,
                "job_description": job_description,
                "messages": []
            }
            agent_outputs = get_agent_outputs(initial_state, graph)

            # Get outputs from resume editor & cover letter generator
            resume_output, cover_letter_output = get_resume_and_coverletter_text_modified(agent_outputs)

            # Extract plain text from output messages
            resume_messages = resume_output.get('messages', [])
            cover_letter_messages = cover_letter_output.get('messages', [])

            resume_text_final = '\n'.join(
                extract_text_from_human_message(msg) for msg in resume_messages if isinstance(msg, HumanMessage)
            ) or "No resume content found."

            cover_letter_text_final = '\n'.join(
                extract_text_from_human_message(msg) for msg in cover_letter_messages if isinstance(msg, HumanMessage)
            ) or "No cover letter content found."

            # Save to DOCX
            resume_doc = save_text_to_docx(resume_text_final, "Tailored Resume")
            cover_letter_doc = save_text_to_docx(cover_letter_text_final, "Cover Letter")

            # Prepare downloads
            resume_file = BytesIO()
            cover_letter_file = BytesIO()
            resume_doc.save(resume_file)
            cover_letter_doc.save(cover_letter_file)
            resume_file.seek(0)
            cover_letter_file.seek(0)

            # Display results
            st.success(" Documents Generated Successfully!")

            st.subheader(" Tailored Resume")
            st.write(resume_text_final)
            st.download_button(
                label="⬇️ Download Tailored Resume",
                data=resume_file,
                file_name="Tailored_Resume.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

            st.subheader(" Cover Letter")
            st.write(cover_letter_text_final)
            st.download_button(
                label="⬇️ Download Cover Letter",
                data=cover_letter_file,
                file_name="Cover_Letter.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

else:
    st.info(" Please upload your resume and enter a job description to begin.")
