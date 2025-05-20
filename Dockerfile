FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
LABEL authors="tjosg"

ENV HF_TOKEN=XXX

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    git htop wget ffmpeg build-essential && \
    rm -rf  /var/lib/apt/lists/*  /var/tmp/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir


RUN huggingface-cli login --token $HF_TOKEN --add-to-git-credential

COPY download_model.py .
COPY generator.py .
COPY models.py .
RUN python download_model.py

COPY . .

CMD ["bash"]