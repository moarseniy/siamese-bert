FROM mlops-docker-local.artifactory.corp.ingos.ru/ai-platform/vllm:0.17.1-cp310-2026_03_31_14_51_59

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    UV_INDEX_URL=https://artifactory.corp.ingos.ru/api/pypi/common-python/simple \
    UV_EXTRA_INDEX_URL=https://artifactory.corp.ingos.ru/api/pypi/mlops-python-local/simple \
    UV_PYPI_TOKEN= \
    UV_NO_PROGRESS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG APP_HOME=/app
ARG APP_USER=10001
ARG APP_GROUP=10001

USER ${APP_USER}
WORKDIR ${APP_HOME}

COPY requirements.txt .
RUN python3 -m venv --system-site-packages venv \
    && . venv/bin/activate \
    && pip install -U pip uv==0.7.21 \
    && uv pip install 'dvc[s3]==3.1.0' -r requirements.txt --no-cache \
        --index-url https://artifactory.corp.ingos.ru/api/pypi/common-python/simple \
        --extra-index-url https://artifactory.corp.ingos.ru/api/pypi/mlops-python-local/simple

COPY app ./app
COPY --chown=${APP_USER}:${APP_GROUP} artifacts ./artifacts
COPY --chmod=0755 start.sh ./start.sh

EXPOSE 8080
STOPSIGNAL SIGTERM

ENTRYPOINT ["/bin/bash"]
CMD ["./start.sh"]
