# Applied Movie Agent system

A multi-agent movie intelligence system built with LangChain and LangGraph. This system orchestrates specialized agents to solve complex, multi-modal research tasks across SQL databases, PDF documents (RAG), and real-time web search.


## Setup & Installation

### 1. Prerequisites
- Python 3.12+
- `uv` (Recommended for high-performance dependency management)
- API Keys: `OPENAI_API_KEY`, `TAVILY_API_KEY`

### 2. Install Dependencies
```bash
uv sync
```

## Project Structure

- `movie_agents.py`: The core logic defining specialists and the supervisor.
- `rest_endpoints.py`: FastAPI server exposing the system via REST.
- `api_models.py`: Pydantic schemas used by FastAPI endpoints
- `chatbot_interface.py`: Gradio web interface for interactive research.
- `main.py`: The single entry point to launch both the API and UI.

## Running the System

To launch the complete integrated system (API + Chatbot):
```bash
uv run python src/main.py
```
- **Web UI**: Access at `http://127.0.0.1:8000/demo`
- **REST Documentation**: Access at `http://127.0.0.1:8000/docs`

## AI Disclosure

To follow the project rules, I am disclosing that I used Claude AI as a technical assistant while building this project. I designed the core logic and the agent system myself, but I used the AI to help with these specific parts:
- `Refining the Prompts`: Wrote the main instructions for the agents, and Claude helped me polish them. It helped me add clear examples so the Supervisor knows how to break down hard questions and which data source to trust most (like trusting the SQL database over a web search).
- `Organizing the Code`: The AI helped me tidy up my code into a cleaner structure. It also helped me write the "error handling" part that prevents the whole system from crashing if a tool or a database query runs into a problem.
- `Chatbot Interface`: I defined the requirements for how the chatbot interface should look. Based on my instructions, Claude provided the code to build the UI.
