FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API script and model
COPY app.py .
COPY unet/ /app/unet/
COPY unet_livecell_best.pth .

# Expose API Port
EXPOSE 8000

# Run FastAPI Server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]