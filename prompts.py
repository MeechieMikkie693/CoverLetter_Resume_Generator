from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def routing_prompt(options, members):
    system_prompt = get_system_prompt()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or is the task complete and should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))

    return prompt

def get_system_prompt():
    return (
        "You are a supervisor tasked with managing a conversation between the "
        "following workers: {members}. The user will provide a job description "
        "and their resume, and wants an edited resume with keywords integrated "
        "based on the job description, and a personalized cover letter. Based on "
        "the conversation so far, respond with the name of the worker who should "
        "act next. When all steps are complete, respond with FINISH."
    )

def get_keyword_generator_agent_prompt():
    return (
        "You are an ATS software. Extract relevant keywords and skills from the job description. "
        "Respond with a list of keywords separated by commas."
    )

def get_resume_generator_agent_prompt():
    return (
        "You are a resume enhancement assistant. Your task is to enhance the user's resume "
        "by integrating relevant keywords and skills. Make sure the enhancements are natural, "
        "professional, and match the job role. In your response, provide only the final resume."
    )

def get_coverletter_generator_agent_prompt():
    return (
        "You are a cover letter generator. Create a personalized cover letter using the job description "
        "and enhanced resume. Focus on highlighting why the candidate is a great fit. In your response, "
        "provide only the final cover letter."
    )
