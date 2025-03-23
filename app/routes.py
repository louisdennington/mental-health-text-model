# Endpoints to handle user input and return suggestions
# Define how the app responds to different URLs (endpoints).
# This is where you say: “When a user visits /, show them the homepage” or “When they submit a form, do this...”

from flask import Blueprint, request, render_template
from app.models.recommend import suggest_resources

bp = Blueprint('main', __name__)

@bp.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["text"]
        suggestions = suggest_resources(user_input)
        return render_template("index.html", suggestions=suggestions)
    return render_template("index.html")
