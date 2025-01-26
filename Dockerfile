# Use a lightweight Python 3.10 base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/tmp/pycharm_project_642 \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_XLA_FLAGS=--tf_xla_enable_xla_devices=0

# Copy requirements.txt first to leverage caching
COPY requirements.txt /tmp/requirements.txt

# Install system dependencies and Python libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server sudo rsync build-essential wget curl git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    apt-get clean && rm -rf /tmp/* /var/tmp/* ~/.cache/pip

# Create a user 'svs' with sudo privileges
RUN useradd -m -s /bin/bash svs && \
    echo 'svs:my001svs314' | chpasswd && \
    usermod -aG sudo svs && \
    mkdir /var/run/sshd && \
    echo "svs ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config

# Create and set ownership of the project directory, data directory
RUN mkdir -p /tmp/pycharm_project_642 && chown svs:svs /tmp/pycharm_project_642
RUN mkdir -p /tmp/data && chown svs:svs /tmp/data

# Define volumes for persistent data
VOLUME ["/tmp/pycharm_project_642", "/tmp/data"]

# Expose ports
EXPOSE 22 8888

# Configure entrypoint and default command
ENTRYPOINT ["bash", "-c"]
CMD ["service ssh start && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]

# создание папки для проекта
# docker volume create my_jupyter_data

# найти ид контейнера
# docker ps
# docker stop my_jupyter_container
# docker rm my_jupyter_container
# docker builder prune --all


# Build the image
# docker build --no-cache -t my_jupyter_image .

# Run the container
# docker run -d --name my_jupyter_container -p 8888:8888 -p 2222:22 -v my_jupyter_data:/tmp/pycharm_project_642 -v data:/tmp/data my_jupyter_image

# docker exec -it my_jupyter_container python -c "import mllab; print(mllab.__file__)"

#{"token": "ya29.a0AXeO80Q2EvzGYd652nR85kBIhwMNqVRQW__7JaowZQwQPzS3PxPBn3R47MnHK_bLuzZOI7VK1v29JpZIsWUpCOUpS1oKonsPqwgdGroeSmvOKj52g-g3JzKXtmz2NC3ZU2sAV4mOPLNvTo44zYehv8hUTz_9Kswp8ylcGEiWaCgYKAWISARISFQHGX2MigPVj-OmYXOeQz616QfoemA0175",
#"refresh_token": "1//0hbXJ5la7UI2YCgYIARAAGBESNwF-L9IrpKXMz87Vd1ig-vpRsmfRGbhI65sk-UNH-2tQGgYLm1W8YTcFtXxkJ9Te9Cw10djXdkQ",
#"token_uri": "https://oauth2.googleapis.com/token",
#"client_id": "1005427861760-o24gdcrjs4mq4nom29evappd9voaj9m8.apps.googleusercontent.com",
#"client_secret": "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",
#"scopes": ["https://www.googleapis.com/auth/drive"],
#"universe_domain": "googleapis.com",
#"account": "", "expiry": "2025-01-26T00:41:53.679406Z"}
