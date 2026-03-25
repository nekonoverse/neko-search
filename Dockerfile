FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME /data
ENV DATA_DIR=/data

# UDS mode: set UDS_PATH (e.g. /var/run/neko-search/neko-search.sock)
# TCP mode (default): listens on port 8002
ENV UDS_PATH=""

EXPOSE 8002

CMD ["sh", "-c", "if [ -n \"$UDS_PATH\" ]; then uvicorn main:app --uds \"$UDS_PATH\"; else uvicorn main:app --host 0.0.0.0 --port 8002; fi"]
