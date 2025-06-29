.PHONY: train serve test format lint clean

# 모델 학습
train:
	python -m scripts.train_model

# 서버 실행
serve:
	python run.py

# 테스트 실행
test:
	pytest tests/

# 코드 포맷팅
format:
	black .
	isort .

# 코드 정적 분석
lint:
	flake8 .

# 정리
clean:
	find . -type d -name "__pycache__" -exec rm -r {} \;
	find . -type d -name ".pytest_cache" -exec rm -r {} \;
	rm -rf .mypy_cache/
