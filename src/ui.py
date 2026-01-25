import gradio as gr
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from src.engine import supervisor


def chat_fn(message, history):
    formatted_messages = []

    # 1. Process history (Gradio 5.x uses list of dicts)
    for entry in history:
        role = entry.get("role")
        content = entry.get("content")
        text = content[0].get("text", "") if isinstance(content, list) else content
        if role == "user":
            formatted_messages.append(HumanMessage(content=text))
        elif role == "assistant":
            formatted_messages.append(AIMessage(content=text))

    # 2. Add current message
    formatted_messages.append(HumanMessage(content=message))

    # 3. Use thread_id for session isolation
    config = {"configurable": {"thread_id": "movie_session_001"}}

    result = supervisor.invoke({"messages": formatted_messages}, config=config)
    return result["messages"][-1].content


# UI Logic for toggling screens
def start_session():
    return gr.update(visible=False), gr.update(visible=True)


with gr.Blocks(title="Movie Agent System") as demo:
    # --- Landing Screen ---
    with gr.Column(visible=True) as landing_screen:
        gr.Markdown("# 🎬 Movie Agent System")
        gr.Markdown("Click the button below to start your research session.")
        start_btn = gr.Button("Start Session", variant="primary")

    # --- Chat Screen (Hidden by default) ---
    with gr.Column(visible=False) as chat_screen:
        # Pre-initialize with your greeting
        chatbot = gr.Chatbot(
            value=[{"role": "assistant", "content": "Hi, I am your movie agent system, how can I help you?"}],
            height=600
        )

        gr.ChatInterface(
            fn=chat_fn,
            chatbot=chatbot,
        )

    # Define the click event to swap screens
    start_btn.click(
        fn=start_session,
        outputs=[landing_screen, chat_screen]
    )

if __name__ == "__main__":
    demo.launch()