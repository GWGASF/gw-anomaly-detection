FROM python:3.10
WORKDIR /opt
SHELL ["/bin/bash", "-c"]

# Install the application dependencies
RUN apt-get update \
#  && apt-get -y install --no-install-recommends vim \
 && apt-get -y upgrade \
 && rm -rf /var/lib/apt/lists/*
 # && pip install --upgrade pip \
 # && pip install poetry

# Install micromamba
RUN mkdir -p /opt/micromamba \
 && cd /opt/micromamba \
 && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \ 
 && export MAMABA_ROOT_PREFIX=/opt/micromamba \
 && eval "$(./bin/micromamba shell hook -s posix)" \
 && ./bin/micromamba shell init -s bash -r /opt/micromamba \
 && source $HOME/.bashrc

# Set conda-forge as the first channel
# RUN /opt/micromamba/bin/micromamba config append channels conda-forge \
 # && /opt/micromamba/bin/micromamba config set channel_priority strict

# Copy in the sourrce code
# COPY . /opt

# Install python packages using poetry
# RUN poetry lock
# RUN poetry install

# Install python Packages using micromamba

# Setup an app user so the container doesn't run as the root user
# RUN useradd app
# USER app

# CMD ["python", "hello.py"]
