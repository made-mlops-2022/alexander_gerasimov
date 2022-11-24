# Online inference

## Installation

### From source

cd online_inference
docker build -t online_inference .

### From DockerHub

docker pull online_inference

## Usage

### Run inference

docker run --rm -p 8000:8000 online_inference               # for local version
docker run --rm -p 8000:8000 alexg25/online_inference_mlops # for DockerHub version

python request_maker.py  # from another terminal

### Run tests

pip install -q pytest pytest-cov
python -m pytest . -v --cov

### Run linter

pip install -q flake8
flake8 --max-line-length=119

## Optimization docker image

By downloading only the necessary packages and enough files to get the job done, docker image can be more than halved. Now the project provides the best option docker image.