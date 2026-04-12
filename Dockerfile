FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

WORKDIR /workspace/vital_lens

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace/vital_lens

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
