FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy env and create it
COPY environment.yaml .
RUN conda update -n base -c defaults conda && \
    conda env create -f environment.yaml && \
    conda clean -afy

# Install framecpp after env creation
RUN conda install -n gwpipelinetest -c conda-forge python-ldas-tools-framecpp -y && \
    conda clean -afy

# Copy code last to leverage layer caching
COPY . .

# Use the correct env name here
ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "gwpipelinetest" ]
CMD [ "bash", "-i" ]
