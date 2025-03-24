from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router
from fastapi.middleware.wsgi import WSGIMiddleware
from starlette.responses import JSONResponse

# Create FastAPI app
fastapi_app = FastAPI()

# CORS settings
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach your routes
fastapi_app.include_router(router)

# WSGI entry point
def application(environ, start_response):
    return WSGIMiddleware(fastapi_app)(environ, start_response)
