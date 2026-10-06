# Python 3.12: audioop is gone in 3.13.
FROM python:3.12-slim@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d
RUN pip install --no-cache-dir wyoming==1.8.0
COPY shim.py /app/shim.py
USER nobody
ENTRYPOINT ["python3", "/app/shim.py"]
