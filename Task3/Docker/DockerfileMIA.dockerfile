# Make sure to use a base image from the official Docker Hub repository with a specific version.
# Visit https://hub.docker.com/ for more information.
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime 
# Install torchvision
RUN pip install torchvision


# Set the working directory to /submission
WORKDIR /submission

# Ensure this is executed in the local pull of the GITHUB directory https://github.com/melanieganz/MIAhackathon2024
# Copy all files into the working directory
COPY ./Task3 /submission/

# Create the necessary directories for loading data into the container.
RUN mkdir /mnt/training_data \
    && mkdir /mnt/training_results \
    && mkdir /mnt/query_data \
    && mkdir /mnt/predicted_data

# Install additional Python dependencies via a requirements file.
# Install any needed packages specified in requirements.txt, added a global requirements.txt outside of the MobileUnetCode folder
COPY ./Task3/Docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# This script assumes that there are two shell Skripts for training and inference.
# Make some adjustments if you are using python scripts for example.
# Ensure the shell scripts are executable.
RUN chmod +x /submission/MobileUnetCode/train.sh /submission/MobileUnetCode/inference.sh

# Uncommend the following command line if custom python steps needs to be 
# exectured during the docker building, for example to download 
# pretrained model parameters or additional configurations. However, we 
# advice to include as much as possible into the copied solution folder or the
# default requirements to be installed by PIP. 
# RUN python /submission/install_solution.py \


# Set the CMD command to run the desired shell script
# Example: run `train.sh`
CMD ["bash", "/submission/MobileUnetCode/train.sh"]
CMD ["bash", "/submission/MobileUnetCode/inference.sh"]