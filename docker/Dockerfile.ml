FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY docker/requirements.ml.txt ./
RUN pip install --no-cache-dir -r requirements.ml.txt

COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY metrics/ metrics/
COPY outputs/ outputs/
COPY mlruns/ mlruns/

CMD ["python", "-m", "src.training"]
