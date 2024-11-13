FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /RLAI

# Copy all the files into the container (including your requirements.txt)
COPY . .

# Install the package manager
RUN conda update -n base -c defaults conda
