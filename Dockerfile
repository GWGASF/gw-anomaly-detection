FROM continuumio/miniconda3:23.5.2-0

# Set working directory
WORKDIR /app

# Copy environment.yaml into the image
COPY environment.yaml .

# Create the environment
RUN conda update -n base -c defaults conda && \
    conda env create -f environment.yaml && \
    conda clean -afy

# Install framecpp separately to avoid solver hang
RUN conda install -n gwtesting -c conda-forge python-ldas-tools-framecpp -y && \
    conda clean -afy

# Copy code after env creation to allow caching
COPY . .

# Force conda shell activation in every command
ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "gwtesting" ]
CMD [ "bash", "-i" ]




# # TEST 1
# FROM python:3.10

# # Set working directory
# WORKDIR /app/gw-anomaly-detection

# # Install dependencies for Poetry and others
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git nano curl \
#     && rm -rf /var/lib/apt/lists/*

# # Install Poetry
# RUN curl -sSL https://install.python-poetry.org | python3 - \
#     && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# # Copy and install dependencies
# COPY pyproject.toml poetry.lock README.md ./
# RUN poetry lock
# RUN poetry install --no-root
# RUN poetry self add poetry-plugin-shell

# # # UNCOMMENT FOR REMOTE REPO
# # # Add the GitLab SSH host key to known_hosts to bypass host key verification prompt
# # RUN mkdir -p ~/.ssh \
# #     && ssh-keyscan -p 30622 gitlab-ssh.nrp-nautilus.io >> ~/.ssh/known_hosts

# # # Clone the GitLab repository into the current working directory using SSH
# # # This allows users to clone the repo using their own SSH key
# # RUN --mount=type=ssh git clone ssh://git@gitlab-ssh.nrp-nautilus.io:30622/gwgasf/gw-anomaly-detection.git .

# # CLONE LOCAL REPO
# # Copy app source
# COPY . .

# # Clean up build tools if desired
# RUN apt-get remove -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# # Default command
# # CMD ["poetry", "run", "python", "gw_anomaly_detection/gasf/src/main.py"]