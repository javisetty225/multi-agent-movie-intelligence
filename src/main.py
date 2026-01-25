import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.server_endpoints import register_chatbot_routes

def create_app() -> FastAPI:
    app = FastAPI(
        title="chatbot",
        description="",
        redirect_slashes=True,
        version="0.0.1",
        openapi_tags=[],
    )

    # Add CORS middleware to allow requests from the website
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_chatbot_routes(app)
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