# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by some Python packages
# (e.g., for cryptography or other C extensions). Add as necessary.
# RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Install pip dependencies
# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./src /app/src

# Copy environment file
COPY .env /app/.env

# Expose the port the app runs on
EXPOSE 3101

# Define the command to run the application
# Run Uvicorn server, listening on 0.0.0.0 to be accessible from outside the container
# --host 0.0.0.0 makes it listen on all available network interfaces
# --port 3101 specifies the port to listen on
# src.main:app points to the FastAPI app instance
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3101"]
