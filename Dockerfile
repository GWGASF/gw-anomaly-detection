FROM python:3.10
WORKDIR /opt

# Install the application dependencies
RUN apt-get update \
# && apt-get -y install --no-install-recommends vim \
 && apt-get -y upgrade \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install poetry

# Copy in the sourrce code
COPY . /opt

# Install python packages
RUN poetry lock
RUN poetry install

# Setup an app user so the container doesn't run as the root user
RUN useradd app
RUN mkdir -p /home/app/data_cache
RUN mkdir -p /home/app/test_data

# CMD ["python", "hello.py"]
