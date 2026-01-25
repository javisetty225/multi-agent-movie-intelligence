from pydantic import BaseModel, ConfigDict, Field


class ChatMessageRequest(BaseModel):
    session_id: str = Field(min_length=1, description="Unique identifier for the chat session.")
    user_query: str = Field(min_length=1, description="User message sent to the chatbot.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "673234d0e59d11f091615228ce713ae0",
                "user_query": "How can I login to my GitHub account?",
            }
        }
    )


class SessionMessageResponse(BaseModel):
    session_id: str = Field(min_length=1, description="Unique identifier for the chat session.")
    chatbot_greeting: str = Field(min_length=1, description="Initial greeting message from the chatbot.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "673234d0e59d11f091615228ce713ae0",
                "chatbot_greeting": "Hi! I'm your assistant. What can I do for you?",
            }
        }
    )


class ChatMessageResponse(BaseModel):
    answer: str = Field(min_length=1, description="Chatbot-generated response.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": (
                    "To log in to your GitHub account, go to the GitHub login page, "
                    "enter your email address and password, and submit the form."
                )
            }
        }
    )
