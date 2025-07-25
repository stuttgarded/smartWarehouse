from flask import Flask
from flask_cors import CORS
import os

def create_app():
    app = Flask(__name__)

    CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

    from .routes.insight_routes import insight
    app.register_blueprint(insight, url_prefix='/api/insight')
    
    return app