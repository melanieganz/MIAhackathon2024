# Make sure to use a base image from the official Docker Hub repository with a specific version.

# source an operating system, we use an official Ubuntu as a parent image 
FROM ubuntu:20.04 
# Install Python 
RUN apt-get update && apt-get install -y python3 python3-pip

# Visit https://hub.docker.com/ for more information.
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime 

# update Python to a fixed version that we used
RUN conda update -n base -c defaults conda && \
    conda install -y python=3.11 && \
    conda update --all --yes

# Set the working directory to /submission
WORKDIR /submission

# Download the code from the GitHub repository
# Install wget with apt-get
RUN apt-get update && apt-get install -y wget unzip
RUN wget https://github.com/melanieganz/MIAhackathon2024/archive/refs/heads/main.zip 
# Unzip the downloaded file
RUN unzip main.zip

# Create the necessary directories for loading data into the container.
RUN mkdir /mnt/training_data \
    && mkdir /mnt/training_results \
    && mkdir /mnt/query_data \  
    && mkdir /mnt/predicted_data
# How do we understand the above folder structure?
# in the above folders the training_data consists of the training data (we assume this contains training_slices and validation_slices),
# the training_results is where one would save the model weights (save the fine-tuned mobilenet here), 
# the query_data is the data to be predicted on aka the test set for us (we assume this contains test_volumes)
# and the predicted_data has the output of our prediction

# Install additional Python dependencies via a requirements file.
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r ./MIAhackathon2024-main/Task3/MobileUnetCode/requirements.txt

# This script assumes that there are two shell Skripts for training and test.
# Make some adjustments if you are using python scripts for example.
# Ensure the shell scripts are executable.
RUN chmod +x ./MIAhackathon2024-main/Task3/MobileUnetCode/train.sh ./MIAhackathon2024-main/Task3/MobileUnetCode/inference.sh

# Set the CMD command to run the desired shell script
# Example: run `train.sh`
CMD ["bash", "./MIAhackathon2024-main/Task3/MobileUnetCode/train.sh"]
CMD ["bash", "./MIAhackathon2024-main/Task3/MobileUnetCode/inference.sh"]