from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router
from fastapi.middleware.wsgi import WSGIMiddleware
from starlette.responses import JSONResponse
from logger import logger

# Create FastAPI app
logger.info("Starting FastAPI application...")
fastapi_app = FastAPI()

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

# WSGI entry point
def application(environ, start_response):
    logger.info("WSGI application triggered.")
    return WSGIMiddleware(fastapi_app)(environ, start_response)
