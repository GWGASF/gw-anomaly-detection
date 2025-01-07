FROM python:3.10
SHELL ["/bin/bash", "-l", "-c"]

# Install the application dependencies
RUN apt-get update \
#  && apt-get -y install --no-install-recommends vim \
 && apt-get -y upgrade \
 && rm -rf /var/lib/apt/lists/*
 # && pip install --upgrade pip \
 # && pip install poetry

# Setup an app user so the container doesn't run as the root user
RUN useradd -ms /bin/bash app
USER app
WORKDIR /home/app/opt

# Install micromamba
RUN cd /home/app \
 && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \ 
 && export MAMABA_ROOT_PREFIX=~/micromamba \
 && eval "$(./bin/micromamba shell hook -s posix)" \
 && ./bin/micromamba shell init -s bash -r /home/app/micromamba \
 && ./bin/micromamba config append channels conda-forge \
 && ./bin/micromamba config set channel_priority strict

# Copy in the sourrce code
RUN mkdir -p /home/app/opt
COPY . /home/app/opt

# Install python packages using poetry
# RUN poetry lock
# RUN poetry install

# Install python Packages using micromamba
RUN /home/app/bin/micromamba create -y -p /home/app/micromamba/env -f /home/app/opt/conda-lock-data.yml

# Add the command to activate the conda environment
RUN echo "micromamba activate /home/app/micromamba/env" >> /home/app/.bashrc

# Set entrypoint to bash
# ENTRYPOINT ["/bin/bash"]
