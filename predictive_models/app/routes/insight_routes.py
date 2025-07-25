from flask import Blueprint, request

insight = Blueprint('insight', __name__)

@insight.route('/try', methods=['GET'])
def try_route():
    return "Hello World"