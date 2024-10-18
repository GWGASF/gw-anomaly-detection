FROM python:3.9
WORKDIR /opt/app

# Install the application dependencies
RUN apt-get update \
 && apt-get -y install --no-install-recommends vim \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install poetry

# Copy in the sourrce code
COPY ./hello.py /opt/app

# Setup an app user so the container doesn't run as the root user
RUN useradd app
USER app

# CMD ["python", "hello.py"]
