FROM python:3.10
SHELL ["/bin/bash", "-l", "-c"]

# Install the application dependencies
RUN apt-get update \
 && apt-get -y upgrade \
 && rm -rf /var/lib/apt/lists/*

# Setup an app user so the container doesn't run as the root user
RUN useradd -ms /bin/bash app

# Copy source code and set ownership in one step
COPY --chown=app:app . /home/app/opt

# Ensure the test_data directory is writable by the app user
RUN mkdir -p /home/app/opt/test_data && chmod -R 777 /home/app/opt/test_data

# Set the working directory
WORKDIR /home/app/opt

# Switch to the non-root user
USER app

# Install micromamba
RUN cd /home/app \
 && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \ 
 && export MAMABA_ROOT_PREFIX=~/micromamba \
 && eval "$(./bin/micromamba shell hook -s posix)" \
 && ./bin/micromamba shell init -s bash -r /home/app/micromamba \
 && ./bin/micromamba config append channels conda-forge \
 && ./bin/micromamba config set channel_priority strict

# Install python Packages using micromamba
RUN /home/app/bin/micromamba create -y -p /home/app/micromamba/env -f /home/app/opt/conda-lock.yml

# Add the command to activate the conda environment
RUN echo "micromamba activate /home/app/micromamba/env" >> /home/app/.bashrc

# Automatically activate Micromamba and run CLI script
CMD ["/bin/bash", "-c", "/home/app/bin/micromamba run -p /home/app/micromamba/env python -u /home/app/opt/cli.py"]
