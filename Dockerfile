FROM python:3.11-slim 

WORKDIR /api

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY /requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api/ .

COPY models/ /models/

EXPOSE 8000

ENV PORT=8000
ENV MODEL_PATH=/model/ann_imdb_final.pt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]