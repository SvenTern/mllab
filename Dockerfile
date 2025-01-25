# Use a lightweight Python 3.10 base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openssh-server \
    build-essential \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter and pip packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir jupyter pandas numpy matplotlib coverage pylint sphinx hudsonthames-sphinx-theme sphinx-rtd-theme releases
RUN pip install --no-cache-dir yfinance mplfinance shimmy
RUN pip install --no-cache-dir polygon-api-client selenium webdriver-manager
RUN pip install --no-cache-dir alpaca-trade-api alpaca-py
RUN pip install --no-cache-dir git+https://github.com/AI4Finance-Foundation/FinRL.git
RUN pip install --no-cache-dir blinker exchange_calendars stockstats stable_baselines3 gym
RUN pip install --no-cache-dir git+https://github.com/SvenTern/mllab.git

# Создайте пользователя svs
RUN useradd -m -s /bin/bash svs

# Настройка SSH
RUN mkdir /var/run/sshd && \
echo 'svs:my001svs314' | chpasswd && \
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config

# Expose ports
EXPOSE 22 8888

# Set up Jupyter Notebook
RUN mkdir -p /root/.jupyter
COPY jupyter_notebook_config.py /root/.jupyter/

# Start SSH and Jupyter Notebook
CMD ["bash", "-c", "service ssh start && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]
