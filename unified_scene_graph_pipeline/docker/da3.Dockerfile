FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/tmp/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ffmpeg git libgl1 libglib2.0-0 ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY third_party/depth_anything_3 /opt/da3
WORKDIR /opt/da3
RUN python -m pip install --upgrade pip \
    && python -m pip install xformers==0.0.28.post3 \
    && python -m pip install -e . \
    && python -m pip install faiss-cpu pandas prettytable numba pypose

COPY da3 /pipeline/da3
WORKDIR /workspace
ENTRYPOINT ["python", "/pipeline/da3/container_scene.py"]
