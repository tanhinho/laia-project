FROM python:3.13-slim

# Install uv
RUN pip install uv

WORKDIR /app
COPY models/* .

# Sync dependencies
RUN uv sync --frozen

# Run the pipeline. 
# We chain the scripts: first download data, then preprocess (generate artifacts) and finally train (log artifacts)
CMD uv run download_data.py && uv run python preprocess.py --train-size 10000 && uv run python linear_regression.py