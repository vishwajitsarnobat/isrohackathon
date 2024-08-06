# isrohackathon
Official Falcon's ISRO Hackathon repo

# Initial Setup
$ git clone https://github.com/vishwajitsarnobat/isrohackathon.git # Cloning the repo will fetch the Dockerfile and project files into the working directory
$ sudo docker build -t isro_hackathon . # Builds the Docker image with the common environment to run the programs
# If above command is throwing error, use { $ systemctl start docker } to start docker service

# Syncing Local Repo with Changes in GitHub Repo
$ git pull origin main # Updates the local repo with the latest changes from the GitHub repo, including the Dockerfile and project files
$ sudo docker build -t isro_hackathon . # Rebuilds the Docker image to update the environment and project files in the container

# To Import Python Libraries or Install Packages via apt
1. Open the Dockerfile and add the necessary commands in the `RUN` section. Combine all previous installation commands into one `RUN` instruction for efficiency.
2. For any new commands or updates, use a separate `RUN` instruction. This helps Docker optimize caching and reduces unnecessary rebuilding. Use chatgpt if you don't know hoe to do this.
$ sudo docker build -t isro_hackathon . # Rebuilds the Docker image with the updated Dockerfile

# Uploading Changes to GitHub Repo
$ git add .
$ git commit -m "Commit message"
$ git push origin main

# Running the Container
$ sudo docker run -it isro_hackathon
$ ls # Lists the project files inside the container

# To clear older unwanted builds
sudo docker container prune -f
sudo docker image prune -f
sudo docker volume prune -f
sudo docker network prune -f

# Steps to Be Usually Followed
1. Initial Setup.
2. Running the Container.
3. Do not write code directly in the container. Modify code on your local system, then rebuild the container to reflect these changes.
4. Include necessary imports and installations in the Dockerfile.
5. Rebuild the container to include new imports and file changes.
6. Use the container to run the code.
7. Use `git add`, `git commit`, and `git push` to update the GitHub repo. Repeat as necessary.

