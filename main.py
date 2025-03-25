import os
from routes import router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from logger import logger

# Create FastAPI app
logger.info("Starting FastAPI application...")
fastapi_app = FastAPI()

# Static hosting for all files in /app, particularly needed for umap_data.json file
fastapi_app.mount("/static", StaticFiles(directory="app"), name="static")

# CORS settings
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.louisdennington.com"],  # Adjust later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured.")

# Attach your routes
fastapi_app.include_router(router)
logger.info("Router attached.")

# Route to serve the HTML interface
@fastapi_app.get("/", response_class=HTMLResponse)
async def serve_ui():
    try:
        with open("app/interface.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        logger.error("interface.html not found at app/interface.html")
        return HTMLResponse(content="<h1>interface.html not found</h1>", status_code=404)

# WSGI entry point
def application(environ, start_response):
    logger.info("WSGI application triggered.")
    return WSGIMiddleware(fastapi_app)(environ, start_response)
