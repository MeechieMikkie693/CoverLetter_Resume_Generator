from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser
from typing import Annotated, Sequence, TypedDict
import operator
import functools

from tools_1 import *
from prompts import *

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def create_agent(llm, tools, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    clean_name = ''.join(e for e in name if e.isalnum() or e in ['_', '-'])
    return {
        "messages": [HumanMessage(content=result["output"], name=clean_name)],
        "next": "supervisor"  # ✅ Required by AgentState
    }

def define_graph(llm):
    members = ["KeyWord Generator", "Resume Editor", "CoverLetter Generator"]
    system_prompt = get_system_prompt()
    options = ["FINISH"] + members

    prompt = routing_prompt(options, members)

    supervisor_chain = (
        prompt
        | llm
        | StrOutputParser()
        | (lambda name: {"next": name})

    )

    keyword_agent = create_agent(llm, [tavily()], get_keyword_generator_agent_prompt())
    keyword_node = functools.partial(agent_node, agent=keyword_agent, name="KeyWord Generator")

    resume_agent = create_agent(llm, [tavily()], get_resume_generator_agent_prompt())
    resume_node = functools.partial(agent_node, agent=resume_agent, name="Resume Editor")

    cover_agent = create_agent(llm, [tavily()], get_coverletter_generator_agent_prompt())
    cover_node = functools.partial(agent_node, agent=cover_agent, name="CoverLetter Generator")

    workflow = StateGraph(AgentState)
    workflow.add_node("KeyWord Generator", keyword_node)
    workflow.add_node("Resume Editor", resume_node)
    workflow.add_node("CoverLetter Generator", cover_node)
    workflow.add_node("supervisor", supervisor_chain)

    for member in members:
        workflow.add_edge(member, "supervisor")

    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END

    workflow.add_conditional_edges("supervisor", lambda x: x["next"].strip(), conditional_map)
    workflow.set_entry_point("supervisor")

    graph = workflow.compile()
    return graph
