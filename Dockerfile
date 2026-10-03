# Use Python slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy backend and frontend code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Expose ports for backend and Jupyter
EXPOSE 8000 8888

# Set default command (can be overridden)
CMD ["bash"]
