import os

# Base upload folder
UPLOAD_FOLDER = "uploads"

# Subfolders
MODEL_FOLDER = os.path.join(UPLOAD_FOLDER, "models")
DATASET_FOLDER = os.path.join(UPLOAD_FOLDER, "datasets")

# Allowed file extensions
ALLOWED_MODEL_EXTENSIONS = {"pkl", "h5", "pt", "joblib", "onnx", "pth"}
ALLOWED_DATASET_EXTENSIONS = {"csv", "jpg", "json", "txt", "jpeg", "png"}