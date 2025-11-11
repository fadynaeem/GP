from flask import Flask, request, jsonify
from controller import AudioAnalysisController
from config import get_config
from req_handling import handling
config = get_config()
app = Flask(__name__)
controller = AudioAnalysisController(config.AUDIO_API_KEY, config.API_KEY)

#//
@app.route('/query', methods=['POST'])
def analyze():
    audio_bytes, error_response, status =handling.get_audio_bytes_from_request(request)
    if error_response:
        return jsonify(error_response), status
    response = controller.handle_query(audio_bytes)
    return jsonify(response)
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT,
            debug=config.FLASK_DEBUG)
# openl3_env\Scripts\activate
    
