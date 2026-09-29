FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. Install System Dependencies & Build Tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ca-certificates \
    curl \
    git \
    wget \
    unzip \
    gnupg \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    android-tools-adb \
    android-tools-fastboot \
    && rm -rf /var/lib/apt/lists/*

# 2. Configure Python Virtual Environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 3. Pre-install Core ML & Quantization Libraries
# Replaced onnxsimplifier with onnxsim
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    "numpy<2.0.0" \
    "torch>=2.0.0" \
    "tensorflow-cpu>=2.10.0,<2.16.0" \
    onnx \
    onnxsim \
    opencv-python \
    jupyter \
    jupyterlab \
    protobuf==3.20.3 \
    flatbuffers \
    sympy \
    scipy \
    matplotlib

# 4. Set Environment Variables expected by QAIRT & Android Toolchains
ENV QAIRT_SDK_ROOT=/opt/qairt
ENV ANDROID_NDK_ROOT=/opt/android-ndk
ENV PATH="${QAIRT_SDK_ROOT}/bin/x86_64-linux-clang:${PATH}"

WORKDIR /workspace

CMD ["/bin/bash"]