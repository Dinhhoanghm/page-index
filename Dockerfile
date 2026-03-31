FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies before copying source (better layer caching)
RUN pip install --no-cache-dir \
    "litellm>=1.82.0" \
    "pymupdf>=1.26" \
    "PyPDF2>=3.0" \
    "pyyaml>=6.0" \
    "anthropic>=0.40" \
    "openai>=1.0" \
    "python-dotenv>=1.0" \
    "fastapi>=0.111" \
    "uvicorn[standard]>=0.29" \
    "python-multipart>=0.0.9"

# Copy application code
COPY api.py ./
COPY PageIndex/ ./PageIndex/

# Runtime directories (mounted as volumes in docker-compose)
RUN mkdir -p workspace uploads

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
