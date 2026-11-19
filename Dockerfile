FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY data ./data
RUN ragpipe ingest data/docs
EXPOSE 8000
CMD ["ragpipe", "serve", "--port", "8000"]
