import logging
import os

import gradio as gr
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
logger = logging.getLogger("MovieAgentUI")


def chat_fn(message, _history, session_id):
    """
    Communicates with the REST API to process the user query.
    The session_id is retrieved from the gr.State component.
    """
    try:
        if not session_id:
            return "No active session found. Please restart."

        payload = {
            "session_id": session_id,
            "user_query": message
        }

        response = requests.post(
            f"{API_BASE_URL}/chat/process_query",
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        data = response.json()
        return data.get("agent_response", "Error: No response from agent.")

    except requests.exceptions.RequestException as e:
        logger.error(f"API Connection Error: {e}")
        return "Failed to connect to the Movie Intelligence API. Is the server running?"


def initialize_ui_session():
    """
    Calls the API to start a new session and prepares the UI transition.
    """
    try:
        response = requests.post(f"{API_BASE_URL}/session/initialize", timeout=10)
        response.raise_for_status()

        data = response.json()
        new_session_id = data["session_id"]
        greeting = data["welcome_message"]

        initial_history = [{"role": "assistant", "content": greeting}]

        return gr.update(visible=False), gr.update(visible=True), new_session_id, initial_history

    except Exception as e:
        logger.error(f"Failed to initialize session via API: {e}")
        return gr.update(), gr.update(), None, [{"role": "assistant", "content": "Backend server unreachable."}]


with gr.Blocks(title=" AppliedAI Movie Agent System") as demo:
    gr.Markdown("# 🎬 AppliedAI Movie Agent System")

    session_id_state = gr.State()

    with gr.Column(visible=True) as landing_screen:
        gr.Markdown("Click the button below to start your cinematic research session.")
        start_btn = gr.Button("Start Session", variant="primary")

    with gr.Column(visible=False) as chat_screen:
        chatbot = gr.Chatbot(
            label="MovieAgent Orchestrator",
            height=800
        )

        gr.ChatInterface(
            fn=chat_fn,
            chatbot=chatbot,
            additional_inputs=[session_id_state]
        )

    start_btn.click(
        fn=initialize_ui_session,
        outputs=[landing_screen, chat_screen, session_id_state, chatbot]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)