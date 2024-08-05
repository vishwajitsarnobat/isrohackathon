# Use the latest Debian image from the Docker Hub
FROM debian:latest

# Update the package list and install basic packages
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        curl \
        vim \
        git \
        neofetch \
        python3-pip \
        python3-venv \
        libproj-dev \
        proj-data \
        proj-bin \
        libgeos-dev

# Create a virtual environment and install Python packages
RUN python3 -m venv /venv
RUN /venv/bin/pip install --upgrade pip
RUN /venv/bin/pip install numpy pandas matplotlib scikit-learn tensorflow xarray netCDF4 cartopy

# Clean up and remove temporary files
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Use the virtual environment in the container
ENV PATH="/venv/bin:$PATH"

# Command to run when the container starts
CMD ["bash"]

