FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/pipeline:/opt/sam3:/opt/sam2 \
    XDG_CACHE_HOME=/tmp/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY third_party/sam3 /opt/sam3
COPY third_party/sam2 /opt/sam2
RUN python -m pip install --upgrade pip \
    && python -m pip install -e /opt/sam3 \
    && SAM2_BUILD_CUDA=0 python -m pip install -e /opt/sam2 \
    && python -m pip install requests==2.32.3 pillow==11.1.0 numpy==1.26.4 \
       opencv-python-headless==4.11.0.86 einops==0.8.1 pycocotools==2.0.10 \
       scipy==1.15.2 pytest==8.3.5
RUN python -c "from sam3.model_builder import build_sam3_video_model; from sam2.build_sam import build_sam2_video_predictor"

COPY pipeline /pipeline
COPY config.json /pipeline/config.json
WORKDIR /workspace
ENTRYPOINT ["python", "/pipeline/cli.py"]
