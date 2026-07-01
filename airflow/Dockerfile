# Image Airflow custom avec les dépendances ML du pipeline Hurdle
FROM apache/airflow:2.9.3-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /tmp/pipeline_requirements.txt
RUN pip install --no-cache-dir -r /tmp/pipeline_requirements.txt
