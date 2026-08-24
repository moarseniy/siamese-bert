ARG VLLM_IMAGE=vllm/vllm-openai:v0.20.1
FROM ${VLLM_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /service

COPY requirements.txt .
RUN python -m venv --system-site-packages venv \
    && . venv/bin/activate \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY start.sh ./start.sh

RUN chmod 0755 start.sh

EXPOSE 8080
STOPSIGNAL SIGTERM

ENTRYPOINT ["/bin/bash", "/service/start.sh"]
