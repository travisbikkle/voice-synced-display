# Base image: CUDA 12.1 + cuDNN 8 + Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python 3.10 and common tools
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-distutils python3-pip git ffmpeg && \
    ln -sf python3.10 /usr/bin/python3 && \
    python3 -m pip install --upgrade pip

# Set workdir
WORKDIR /app

# Copy project files
COPY . /app

# Create virtual environment
RUN python3 -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install non-torch dependencies using China mirror
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# Install CUDA version of torch (for GPU)
RUN pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# Expose port
EXPOSE 8000

# Start the app
CMD ["python", "run.py"] 