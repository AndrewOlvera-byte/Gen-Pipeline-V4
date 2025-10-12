ARG BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    TORCH_ALLOW_TF32_CUBLAS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      libgl1-mesa-glx libglfw3 libosmesa6 libglew-dev patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps first for build cache
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# project code
COPY . /app

# default command: training entry; Compose or CLI can override freely
CMD ["python", "scripts/train.py"]
