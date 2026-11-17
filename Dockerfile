FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /project

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/processed/ ./data/processed/
COPY vector_db/structured/ ./vector_db/structured/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["python", "-m", "streamlit", "run", "app/ui.py",      "--server.address=0.0.0.0",      "--server.port=8501",      "--server.headless=true"]
