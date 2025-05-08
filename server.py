from flask import Flask
from flask_cors import CORS
from routes import register_routes
from model_loader import tokenizer, model, DEVICE
from db_pool import db_pool
import os
import sys

try:
    print("🚀 Starting Flask server...")
    app = Flask(__name__)
    CORS(app)

    try:
        register_routes(app)
        print("✅ Routes registered successfully.")
    except Exception as e:
        print(f"❌ Failed to register routes: {e}")
        sys.exit(1)

    if __name__ == "__main__":
        port = int(os.getenv("PORT", 4000))
        try:
            app.run(host="0.0.0.0", port=port)
        except Exception as e:
            print(f"❌ Failed to run Flask app: {e}")
            sys.exit(1)

except Exception as e:
    print(f"❌ Uncaught error in server startup: {e}")
    sys.exit(1)
