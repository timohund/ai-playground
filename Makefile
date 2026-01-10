all: build up ollama-install-llama3 ollama-install-tinyllama run

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
	docker compose exec ollama ollama pull llama3.1:8b

ollama-rm-llama3:
	docker compose exec ollama ollama rm llama3.1:8b

ollama-install-deepseek-r1:
	docker compose exec ollama ollama pull deepseek-r1:32b

ollama-rm-deepseek-r1:
	docker compose exec ollama ollama rm deepseek-r1:32b

ollama-install-mistral-3:
	docker compose exec ollama ollama pullm mistral-small3.2:24b

ollama-rm-mistral-3:
	docker compose exec ollama ollama rm mistral-small3.2:24b

ollama-install-tinyllama:
	docker compose exec ollama ollama pull tinyllama

ollama-rm-tinyllama:
	docker compose exec ollama ollama rm tinyllama
