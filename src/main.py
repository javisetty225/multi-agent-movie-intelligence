import logging

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.rest_endpoints import register_movie_intelligence_routes
from src.frontend.chatbot_interface import demo


def create_app() -> FastAPI:
    app = FastAPI(
        title="chatbot",
        description="",
        redirect_slashes=True,
        version="0.0.1",
        openapi_tags=[],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_movie_intelligence_routes(app)
    gr.mount_gradio_app(app, demo , path="/demo")

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    app = create_app()

    uvicorn.run(
        app,
        workers=1,
    )


if __name__ == "__main__":
    main()