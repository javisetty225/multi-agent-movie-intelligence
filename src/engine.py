import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import SQLDatabase
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. SETUP CORE COMPONENTS
llm = ChatOpenAI(model="gpt-4o", temperature=0)

db_file_path = Path("data/SQL_Movies.db")
db = SQLDatabase.from_uri("sqlite:///" + str(db_file_path.resolve()))

# Standardize the RAG retriever
rag_movie_pdf_path = Path("data/RAG_movies.pdf")
loader = PyPDFLoader(str(rag_movie_pdf_path.resolve()))
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    add_start_index=True
)
pages = loader.load()
splits = text_splitter.split_documents(pages)
vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 2. DEFINE SPECIALIZED WORKER AGENTS
# SQL Worker: Uses a Toolkit to check schema before querying (prevents OperationalError)
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_worker = create_agent(
    llm,
    tools=sql_toolkit.get_tools(),
    system_prompt=(
        "You are a SQL expert. Before querying, ALWAYS list tables and "
        "check the schema of relevant tables to ensure column names are correct. "
        "If a query fails, check the schema and try a corrected version."
    )
)

# RAG Worker: Named tool for internal document search
@tool
def search_pdf_docs(query: str) -> str:
    """Search internal PDF documents for movie plots, genres, and descriptions."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

rag_worker = create_agent(
    llm,
    tools=[search_pdf_docs],
    system_prompt="You are a research assistant. Use the search tool to answer questions about plots/genres."
)

# Research Worker: Live web search
search_worker = create_agent(
    llm,
    tools=[TavilySearch(k=3)],
    system_prompt="You are a web researcher. Find external info like awards or current box office data."
)

# 3. WRAP WORKERS AS TOOLS FOR THE SUPERVISOR
@tool
def call_sql_agent(query: str, config: RunnableConfig):
    """Use for quantitative data: counts, years, ratings, or specific database facts."""
    response = sql_worker.invoke({"messages": [HumanMessage(content=query)]}, config)
    return response["messages"][-1].content

@tool
def call_rag_agent(query: str, config: RunnableConfig):
    """Use for qualitative data: plots, themes, and movie descriptions from internal documents."""
    response = rag_worker.invoke({"messages": [HumanMessage(content=query)]}, config)
    return response["messages"][-1].content

@tool
def call_research_agent(query: str, config: RunnableConfig):
    """Use for external information: producers, actors, awards, or any web-based data."""
    response = search_worker.invoke({"messages": [HumanMessage(content=query)]}, config)
    return response["messages"][-1].content

# 4. BUILD THE COLLABORATIVE SUPERVISOR
tools = [call_sql_agent, call_rag_agent, call_research_agent]
supervisor = create_agent(
    llm,
    tools=tools,
    system_prompt=(
        "You are the Movie System Supervisor. Your goal is to answer user questions "
        "by coordinating between your specialist agents. If a question has multiple parts, "
        "call the relevant agents in sequence to gather all information "
        "before providing a final, comprehensive summary to the user."
    ),
    checkpointer=MemorySaver()
)