FROM mlops-docker-local.artifactory.corp.ingos.ru/ai-platform/vllm:0.17.1-cp310-2026_03_31_14_51_59

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG APP_HOME=/app
ARG APP_USER=10001
ARG APP_GROUP=10001

USER ${APP_USER}
WORKDIR ${APP_HOME}

COPY --chown=${APP_USER}:${APP_GROUP} artifacts ./artifacts
COPY --chmod=0755 start.sh ./start.sh

EXPOSE 8000
STOPSIGNAL SIGTERM

ENTRYPOINT ["/bin/bash"]
CMD ["./start.sh"]
