FROM python:3.11-slim

# Create non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 appuser
ENV HOME=/home/appuser
ENV PATH=/home/appuser/.local/bin:$PATH

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
