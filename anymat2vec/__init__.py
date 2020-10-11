import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
