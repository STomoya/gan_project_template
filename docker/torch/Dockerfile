
ARG PYTORCH_VERSION=1.11.0
ARG PYTORCH_CUDA_VERSION=11.3
ARG PYTORCH_CUDNN_VERSION=8

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${PYTORCH_CUDA_VERSION}-cudnn${PYTORCH_CUDNN_VERSION}-devel

# erase CUDA from apt list. This is temporal, until PyTorch update it's official images.
# See https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# apt installs
ENV DEBIAN_FRONTEND noninteractive
RUN apt update -y && apt install -y \
    libopencv-dev \
    git

# user
ARG UID=1000
ARG USERID=torchuser
RUN useradd -l -m -u ${UID} ${USERID}
USER ${UID}
ENV PATH="/home/${USERID}/.local/bin:${PATH}"

# install pip modules
COPY ./docker/torch/requirements.txt requirements.txt
RUN pip --default-timeout=100 install --user -r requirements.txt
