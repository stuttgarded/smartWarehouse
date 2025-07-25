from flask import Blueprint, request

from app.controllers.insight_controller import InsightController

insight = Blueprint('insight', __name__)

@insight.route('/sales', methods=['POST'])
def try_route():
    return InsightController.process(request)