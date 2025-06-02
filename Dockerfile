FROM python:3.12

# Set working directory
WORKDIR /main

# Copy everything into the container
COPY . .

# Print files to verify config/config.yaml is copied correctly
RUN echo "🟢 Listing files to confirm config/config.yaml is there:" && \
    ls -R /main

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Set environment variables (like PYTHONUNBUFFERED to show logs immediately)
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8000

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
