import logging
import os
from pathlib import Path
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_chroma import Chroma
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MovieAgentSystem")


class ToolErrorMiddleware(AgentMiddleware):
    """
    Middleware to handle tool failures and provide error feedback to the LLM.
    """

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable
    ) -> ToolMessage:
        """
        Intercepts tool calls to catch exceptions.

        Args:
            request: The tool call request object.
            handler: The function that executes the tool.
        """
        try:
            logger.info(f"Invoking tool: {request.tool_call['name']}")
            return handler(request)
        except Exception as e:
            logger.error(f"Tool {request.tool_call['name']} failed: {str(e)}")
            return ToolMessage(
                content=f"Error: The tool failed with: {str(e)}. Please check your inputs.",
                tool_call_id=request.tool_call["id"]
            )


handle_tool_errors = ToolErrorMiddleware()


def initialize_resources() -> tuple[SQLDatabase, VectorStoreRetriever]:
    """
    initializes the relational DB and Vector store.

    Returns:
        A tuple containing the initialized SQLDatabase and VectorStoreRetriever.
    """
    PROJECT_ROOT = Path(__file__).parent.parent
    DB_PATH = PROJECT_ROOT / "data" / "SQL_Movies.db"
    PDF_PATH = PROJECT_ROOT / "data" / "RAG_movies.pdf"
    CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH.resolve()}")

    embeddings = OpenAIEmbeddings()

    if CHROMA_PATH.exists():
        logger.info("Loading existing vector store from disk...")
        vectorstore = Chroma(
            persist_directory=str(CHROMA_PATH.resolve()),
            embedding_function=embeddings
        )
    else:
        logger.info("No vector store found. Creating new index from PDF...")
        if not PDF_PATH.exists():
            raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

        loader = PyPDFLoader(str(PDF_PATH.resolve()))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1100,
            chunk_overlap=150,
            add_start_index=True
        )
        splits = text_splitter.split_documents(loader.load())
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(CHROMA_PATH.resolve())
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return db, retriever


def create_movie_intelligence_system():
    """
    Constructs the multi-agent hierarchy with specialized workers and a supervisor.

    Returns:
        A compiled LangGraph object (the Supervisor).
    """
    db, retriever = initialize_resources()
    llm = ChatOpenAI(model="gpt-4o", temperature=0, timeout=60, api_key=os.getenv("OPENAI_API_KEY"))

    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_worker = create_agent(
        llm,
        tools=sql_toolkit.get_tools(),
        system_prompt=(
            "You are a SQL Database Specialist. Your role is to perform accurate analytics and structured queries. "
            "CRITICAL GUIDELINES: 1. ALWAYS run 'sql_db_schema' at start. 2. TABLE RELATIONSHIPS: movies↔movie_actor via movie_id, movies↔languages via language_id, movies↔financials via movie_id. "
            "3. QUERY OPTIMIZATION: Use WHERE for filtering, GROUP BY for aggregations, HAVING for filtering results, ORDER BY LIMIT for rankings. "
            "4. ERROR RECOVERY: If match fails, run diagnostic queries (SELECT DISTINCT language_name FROM languages, SELECT DISTINCT genre FROM movies). Retry with normalized terms. "
            "5. ACCURACY: Return exact counts and include column names."
        ),
        middleware=[handle_tool_errors]
    )

    @tool
    def search_pdf_docs(query: str) -> str:
        """Search internal PDF documents for movie plots and descriptions."""
        docs = retriever.invoke(query)

        if not docs:
            return f"No movie plots or descriptions found for query: '{query}'."

        return "\n\n".join([d.page_content for d in docs])

    rag_worker = create_agent(
        llm,
        tools=[search_pdf_docs],
        system_prompt=(
            "You are a Document Researcher specializing in semantic search and movie identification. "
            "KEY RESPONSIBILITIES: 1. PLOT SEARCH: Search by plot points, themes, characters. Return 2-3 most relevant with context. "
            "2. MOVIE IDENTIFICATION: For plot descriptions (e.g., 'Project Mayhem'), identify EXACT official 'Movie Name' from PDF. Include confidence level. "
            "3. GENRE MATCHING: Extract and confirm all genres clearly. "
            "4. AMBIGUITY: If multiple matches, return all with distinguishing details. If no exact match, provide best partial matches. "
            "5. QUALITY: Verify titles match official names, cross-reference genre/year."
        ),
        middleware=[handle_tool_errors]
    )

    search_worker = create_agent(
        llm,
        tools=[TavilySearch(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))],
        system_prompt=(
            "You are a Web Researcher specializing in current movie industry information. "
            "SPECIALIZATION: 1. PRODUCERS & DIRECTORS: Search filmography, verify accuracy, return exact credited names. "
            "2. AWARDS: Find Oscar, Golden Globe, Cannes, BAFTA nominations/wins with categories and years. Verify against official sources. "
            "3. CAST & CREW: Retrieve detailed lists with roles and birth names if pseudonyms used. "
            "4. VERIFICATION: Cross-reference multiple sources, flag conflicts, prioritize official sources over fan sites. "
            "5. SCOPE: Focus only on data NOT in local SQL database. Verify movie title matches before searching."
        ),
        middleware=[handle_tool_errors]
    )


    @tool
    def call_sql_agent(query: str, config: RunnableConfig) -> str:
        """Use for structured queries: counts, rankings, financials, and language-specific filters."""
        res = sql_worker.invoke({"messages": [HumanMessage(content=query)]}, config)
        return res["messages"][-1].content

    @tool
    def call_rag_agent(query: str, config: RunnableConfig) -> str:
        """Use for story summaries, identifying film titles from plots, and genres."""
        res = rag_worker.invoke({"messages": [HumanMessage(content=query)]}, config)
        return res["messages"][-1].content

    @tool
    def call_research_agent(query: str, config: RunnableConfig) -> str:
        """Use for finding Producers, Directors, and news not in our local files."""
        res = search_worker.invoke({"messages": [HumanMessage(content=query)]}, config)
        return res["messages"][-1].content

    return create_agent(
        llm,
        tools=[call_sql_agent, call_rag_agent, call_research_agent],
        system_prompt=(
            "You are the MovieAgent Orchestrator coordinating specialized agents. "
            "DECOMPOSITION: Analyze question type (structured/descriptive/external). Break multi-part questions into steps. Plan parallel vs sequential calls. "
            "EXECUTION ORDER: START RAG if plot/theme described; START SQL if counts/dates/ratings/direct names; START Web for producers/directors/awards/industry info. "
            "WORKFLOWS: (1) 'How many 1999 movies?' → SQL only. (2) 'Titanic about?' → RAG only. (3) 'Project Mayhem plot, release, producers?' → RAG→SQL→Web sequential. (4) 'Sci-fi 90s >8.0 rating?' → SQL then RAG verification. "
            "ERROR HANDLING: No RAG results→try SQL differently; SQL fails→identify via RAG first; Web empty→return local data with note; Multiple matches→narrow with criteria. "
            "CONFLICT RESOLUTION: Priority=Local DB > PDF > Web. Report source when conflicts exist. "
            "SYNTHESIS: Combine findings into ONE clear answer. Include source attribution. For complex queries, summarize workflow used. Flag incomplete results."
        ),
        checkpointer=MemorySaver()
    )

supervisor = create_movie_intelligence_system()