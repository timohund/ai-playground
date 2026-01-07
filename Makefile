all: build up ollama-install-gemini-flash run

build:
	docker compose build

up:
	docker compose up -d

recreate:
	docker compose up --force-recreate

down:
	docker compose down

run:
	docker compose exec dspy python app.py

ollama-list:
	docker compose exec ollama ollama list

ollama-install-llama3:
	docker compose exec ollama ollama pull llama3

ollama-rm-llama3:
	docker compose exec ollama ollama rm llama3

ollama-install-gemini-flash:
	docker compose exec ollama ollama pull gemini-3-flash-preview

ollama-rm-gemini-flash:
	docker compose exec ollama ollama rm gemini-3-flash-preview

ollama-install-tinyllama:
	docker compose exec ollama ollama pull tinyllama

ollama-rm-tinyllama:
	docker compose exec ollama ollama rm tinyllama

