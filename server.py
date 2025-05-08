from flask import Flask
from flask_cors import CORS
from routes import register_routes
from model_loader import tokenizer, model, DEVICE
from db_pool import db_pool
import os

app = Flask(__name__)
CORS(app)
register_routes(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 4000)))
