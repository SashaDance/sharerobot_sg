FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/pipeline:/opt/robotseg/test \
    XDG_CACHE_HOME=/tmp/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ffmpeg libgl1 libglib2.0-0 ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY third_party/robotseg /opt/robotseg
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-build-isolation -e /opt/robotseg \
    && python -m pip install opencv-contrib-python-headless==4.11.0.86 \
       scipy==1.15.2 matplotlib==3.10.1 natsort==8.4.0 \
    && cd /opt/robotseg && TORCH_CUDA_ARCH_LIST=8.0 ROBOTSEG_BUILD_ALLOW_ERRORS=0 python setup.py build_ext --inplace \
    && python -c "from robotseg.build_robotseg import build_robotseg_video_predictor; from utils import guided_refine_mask"

COPY pipeline /pipeline
COPY robot_tracking_config.json /pipeline/robot_tracking_config.json
WORKDIR /workspace
ENTRYPOINT ["python", "/pipeline/robot_tracking_compare.py"]
