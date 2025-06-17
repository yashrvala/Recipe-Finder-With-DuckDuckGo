import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import SystemMessage
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

st.title("🍳 Recipe Finder with DuckDuckGo")
query = st.text_input("Enter your recipe query (e.g., 'How to make butter paneer tikaa'): ")

if query:
    # Initialize DuckDuckGo tool
    search_tool = DuckDuckGoSearchRun()

    # Tool setup for agent
    tools = [
        Tool(
            name="DuckDuckGoSearch",
            func=search_tool.run,
            description="Search the web for recipe data"
        )
    ]

    # Set up LLM
    llm = AzureChatOpenAI(
    openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_name="gpt-4o",
    temperature=0.7,
)
    
    system_message = SystemMessage(
        content="You are a helpful cooking assistant that provides clear, step-by-step recipes. Always include ingredients, preparation steps, and tips if available. dont give answer of out of this concept"
    )

    # Create agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )

    # Run the agent
    with st.spinner("🔍 Searching for recipe..."):
        result = agent.run(query)

    # Show result
    st.subheader("🥘 Recipe Result")
    st.write(result)
