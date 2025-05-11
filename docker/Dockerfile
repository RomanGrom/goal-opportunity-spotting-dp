# syntax=docker/dockerfile:1
FROM vggfiit/vgg-torch:latest
#FROM nvidia/opengl:1.0-glvnd-runtime


# Update the APT-GET
RUN apt-get update --fix-missing
RUN apt-get install apt-transport-https
RUN apt-get install -y ca-certificates


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -yq git cmake build-essential \
  libgl1-mesa-dev libsdl2-dev \
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
  libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip


USER root

# Install required packages
WORKDIR /setup
COPY requirements.txt requirements.txt

# Install specific version of setuptools
RUN pip install setuptools==58.0.4

RUN pip install -r requirements.txt

RUN pip install --upgrade typing_extensions
RUN pip install fastdtw
RUN pip install scikit-learn

WORKDIR /workspace/goal_spotting