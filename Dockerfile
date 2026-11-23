# Formica control panel: Python (standard library) + docker CLI with compose.
# It manages the other containers through the Docker socket of the host.
FROM docker:cli
RUN apk add --no-cache python3 tzdata git
COPY VERSION /app/VERSION
COPY control/ /app/
ENV FORMICA_HOME=/opt/formica \
    FORMICA_PORT=8095 \
    PYTHONUNBUFFERED=1
EXPOSE 8095
ENTRYPOINT ["python3", "/app/formica.py"]
CMD ["serve"]
