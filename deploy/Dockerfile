# Grey Box monitoring API. CPU image by default (text monitor, real time).
#   docker build -t greybox -f deploy/Dockerfile .
#   docker run -p 8000:8000 -v $PWD/greybox-text-monitor.json:/models/text.json -e GB_TEXT_MONITOR=/models/text.json greybox
# GPU reader monitor: build with --build-arg READER=1 and run with --gpus all -e GB_MONITOR_PROBE=/models/reader_probe.joblib
FROM python:3.11-slim
ARG READER=0
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[server]" \
 && if [ "$READER" = "1" ]; then pip install --no-cache-dir torch transformers huggingface_hub joblib scikit-learn; fi
EXPOSE 8000
CMD ["uvicorn", "graybox.monitor_server:app", "--host", "0.0.0.0", "--port", "8000"]
