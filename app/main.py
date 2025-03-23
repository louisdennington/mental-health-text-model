# Flask or FastAPI app entry point

from flask import Flask
from app.routes import bp

# Import the app
app = Flask(__name__)
app.register_blueprint(bp)

# Define how it runs (app.run())

# Link routes from routes.py

if __name__ == "__main__":
    app.run(debug=True)