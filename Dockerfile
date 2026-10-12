# TFG-Net one-command reproduction image
# Paper environment: PyTorch 2.1 + CUDA 12.1, Ultralytics 8.1 (RTX 4090).
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for OpenCV / pycocotools
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace
RUN pip install --no-cache-dir -e .

# Make the synthetic sample data + run a CPU smoke test at build time
RUN python scripts/make_fake_data.py --out data/fake --n 16 \
    && python -m pytest -q tests -m "not slow" || true

ENTRYPOINT ["python"]
CMD ["scripts/train.py", "--help"]
