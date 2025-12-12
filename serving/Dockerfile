FROM python:3.13-slim

# Install uv
RUN pip install uv

WORKDIR /app
COPY serving/* .

# Sync dependencies
RUN uv sync --frozen

# Expose FastAPI port
EXPOSE 8080

# Start App
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]