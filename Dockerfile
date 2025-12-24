FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install essential packages and dependencies needed for Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    gnupg \
    ca-certificates \
    procps \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN pip install --no-cache-dir -r requirements.txt
RUN patchright install chromium --with-deps

# Copy the rest of the application
COPY . .

# Create a data directory with proper permissions
RUN mkdir -p /app/data && chmod 777 /app/data
RUN mkdir -p /app/media && chmod 777 /app/media

# Expose the port the app runs on
EXPOSE 8000

# Create a non-root user to run the app
RUN adduser --disabled-password --gecos "" appuser
# Give appuser permissions to the necessary directories
RUN chown -R appuser:appuser /app

USER appuser

# Set healthcheck to ensure the service is running properly
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8000/api/v1/ping || exit 1

# Command to run the application
CMD ["python", "app.py"] 