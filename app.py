import os
import sys
from flask import Flask, jsonify
from Config import Config
from controller import verse_bp
from CustomJSONEncoder import CustomJSONEncoder

os.environ["KMP_DUPLICATE_LIB_OK"] = Config.KMP_DUPLICATE_LIB_OK

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.json_encoder = CustomJSONEncoder
    app.register_blueprint(verse_bp)

    @app.route('/')
    def index():
        return jsonify({
            "status": "running",
            "message": (
                "Quran Verse API is running. Use /stats or /query_audio "
                "endpoints."
            )
        })
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def server_error(error):
        return jsonify({"error": "Internal server error"}), 500
    return app


def check_environment():
    required_vars = ["PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(
            "Error: The following required environment variables are not set: "
            f"{', '.join(missing_vars)}"
        )
        print(
            "Please set these variables in your environment or create a .env "
            "file."
        )
        print("See .env.example for a template.")
        return False
    return True


if __name__ == '__main__':
    if not check_environment():
        sys.exit(1)
    app = create_app()
    print(f"Starting server on {Config.HOST}:{Config.PORT}")
    print(f"Debug mode: {Config.DEBUG}")
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )