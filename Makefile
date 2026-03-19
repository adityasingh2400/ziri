.PHONY: install run test lint docker-up start

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	python -m compileall app tests

docker-up:
	docker compose up --build

start:
	@echo "Starting Ziri (server + listener)..."
	@echo "  Dashboard:  http://localhost:8000/listen"
	@echo "  Status:     http://localhost:8000/status"
	@echo "  Metrics:    http://localhost:8000/metrics"
	@(sleep 2 && open http://localhost:8000/listen) &
	./.venv/bin/python run_listener.py --no-vision
