import logging
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from src.engine import supervisor
from src.server_schemas import ChatMessageRequest, SessionMessageResponse, ChatMessageResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Movie Agent API")


def register_chatbot_routes(app: FastAPI):
    """
    Register chatbot-related REST endpoints on the FastAPI app.
    """

    @app.post("/session", response_model=SessionMessageResponse)
    async def create_session():
        """
        Starts a new session. Generates a thread_id and returns the greeting.
        """
        try:
            session_id = str(uuid4())
            greeting = "Hi, I am your movie agent system, how can I help you?"
            
            logger.info(f"Session started: {session_id}")
            return SessionMessageResponse(session_id=session_id, chatbot_greeting=greeting)
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise HTTPException(status_code=500, detail="Could not initialize session")

    @app.post("/chat", response_model=ChatMessageResponse)
    async def chat(request: ChatMessageRequest):
        """
        Processes a message for an existing session. 
        LangGraph handles the history automatically using session_id as thread_id.
        """
        try:
            # Use session_id as the thread_id for persistence
            config = {"configurable": {"thread_id": request.session_id}}
            
            # Invoke supervisor. LangGraph loads history from MemorySaver automatically.
            # We only pass the NEW HumanMessage.
            result = await supervisor.ainvoke(
                {"messages": [HumanMessage(content=request.user_query)]}, 
                config=config
            )
            
            # The final message in the state is the Supervisor's summary
            final_answer = result["messages"][-1].content
            
            return ChatMessageResponse(answer=final_answer)
        except Exception as e:
            logger.error(f"Error in chat session {request.session_id}: {e}")
            raise HTTPException(status_code=500, detail="Error processing your request")