from pydantic import BaseModel, ConfigDict, Field


class MovieQueryRequest(BaseModel):
    session_id: str = Field(min_length=1, description="Unique identifier for the chat session.")
    user_query: str = Field(min_length=1, description="User message sent to the chatbot.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "673234d0e59d11f091615228ce713ae0",
                "user_query": "Who directed Inception?",
            }
        }
    )


class SessionInitializationResponse(BaseModel):
    session_id: str = Field(min_length=1, description="Unique identifier for the chat session.")
    welcome_message: str = Field(min_length=1, description="Initial greeting message from the chatbot.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "673234d0e59d11f091615228ce713ae0",
                "welcome_message": "Hi! I'm your assistant. What can I do for you?",
            }
        }
    )


class MovieQueryResponse(BaseModel):
    agent_response: str = Field(min_length=1, description="Chatbot-generated response.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "agent_response": (
                    "The movie 'Inception' was directed by Christopher Nolan."
                )
            }
        }
    )
