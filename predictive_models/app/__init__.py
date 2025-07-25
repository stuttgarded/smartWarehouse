from flask import Flask
import os

def create_app():
    app = Flask(__name__)

    from .routes.insight_routes import insight
    app.register_blueprint(insight, url_prefix='/api/insight')
    
    return app