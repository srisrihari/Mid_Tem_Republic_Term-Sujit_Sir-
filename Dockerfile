# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/uploads && \
    mkdir -p /app/models && \
    mkdir -p /app/data && \
    chmod 777 /app/uploads && \
    chmod 777 /app/models && \
    chmod 777 /app/data

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Make port 5001 available to the world outside this container
EXPOSE 5001

# Run the application with host 0.0.0.0 to make it accessible outside container
CMD ["python", "src/flask_app.py"]
