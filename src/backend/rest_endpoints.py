import logging
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage

from src.backend.api_models import MovieQueryRequest, MovieQueryResponse, SessionInitializationResponse
from src.backend.movie_agents import supervisor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MovieIntelligenceAPI")


def register_movie_intelligence_routes(api_app: FastAPI) :
    """
    Registers intent-driven REST endpoints for the movie intelligence system.
    """

    @api_app.post("/session/initialize", response_model=SessionInitializationResponse)
    async def initialize_chat_session():
        """
        Generates a unique conversation ID and provides the initial bot greeting.
        """
        try:
            unique_session_id = str(uuid4())
            initial_greeting = "Hi, I am your movie agent system, how can I help you?"

            logger.info(f"Initialized new MovieAgent session: {unique_session_id}")

            return SessionInitializationResponse(
                session_id=unique_session_id,
                welcome_message=initial_greeting
            )
        except Exception as e:
            logger.error(f"Critical failure during session initialization: {e}")
            raise HTTPException(status_code=500, detail="Internal server error occurred")

    @api_app.post("/chat/process_query", response_model=MovieQueryResponse)
    async def process_movie_query(query_request: MovieQueryRequest):
        """
        Orchestrates specialized agents to answer a query.
        Uses the provided session_id to maintain conversational continuity.
        """
        try:
            persistence_config = {"configurable": {"thread_id": query_request.session_id}}

            logger.info(f"Processing query for session {query_request.session_id}")

            # Invoke the supervisor; MemorySaver handles state lookup automatically
            orchestrator_output = await supervisor.ainvoke(
                {"messages": [HumanMessage(content=query_request.user_query)]},
                config=persistence_config
            )

            final_answer = orchestrator_output["messages"][-1].content

            return MovieQueryResponse(agent_response=final_answer)

        except Exception as e:
            logger.error(f"Error during query processing for session {query_request.session_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail="The movie system encountered an error while processing your query."
            )
