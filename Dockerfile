# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir git+https://github.com/openai/whisper.git \
    numpy \
    torch

# Copy the benchmarking script into the container
COPY benchmark_whisper.py .

# Set the entrypoint to run the script
ENTRYPOINT ["python", "benchmark_whisper.py"]

# Default command (can be overridden)
CMD ["base", "/audio/input.mp3"]
