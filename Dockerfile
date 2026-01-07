FROM python:3.10

WORKDIR /app

COPY app.py /app/

RUN pip install --upgrade pip

RUN pip install ollama mlflow dspy-ai

CMD ["tail", "-f", "/dev/null"]
