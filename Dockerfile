# hardware requirements: 4 NVIDIA GPUs supporting CUDA with high enough GPU RAM
FROM python:3.8
# TODO: base image: Python 3.8.0 + NVIDIA driver + Cudnn + CUDA 11.2

RUN pip install -r requirements.txt

RUN python -m spacy download de && python -m spacy download en