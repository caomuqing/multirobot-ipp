# COMPASS - Ubuntu 22.04 + ROS2 Humble + Python
FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Apt flags for reliability
ARG UPDATE_FLAGS="-o Acquire::Retries=3 -o Acquire::http::Timeout=120"
ARG INSTALL_FLAGS="-o Acquire::Retries=3 -o Acquire::http::Timeout=120 -o Dpkg::Options::=--force-confdef"

# System deps
RUN apt-get ${UPDATE_FLAGS} update -y \
  && apt-get ${INSTALL_FLAGS} install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg2 \
    lsb-release \
    software-properties-common \
    python3-pip \
    python3.10 \
    python3.10-venv \
    python3-dev \
    git \
  && rm -rf /var/lib/apt/lists/*

# Install ROS2 Humble
RUN add-apt-repository universe \
  && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
  && apt-get ${UPDATE_FLAGS} update -y \
  && apt-get ${INSTALL_FLAGS} install -y --no-install-recommends \
    ros-humble-desktop \
    python3-argcomplete \
  && rm -rf /var/lib/apt/lists/*

# ROS2 environment
ENV ROS_DISTRO=humble
ENV AMENT_PREFIX_PATH=/opt/ros/humble
ENV COLCON_PREFIX_PATH=/opt/ros/humble
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib
ENV PATH=/opt/ros/humble/bin:$PATH
ENV PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages
ENV ROS_PYTHON_VERSION=3
ENV ROS_VERSION=2
ENV ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
ENV DEBIAN_FRONTEND=

# Source ROS2 in shell
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

# Create venv to avoid conflicts with ROS2 apt-installed packages (e.g. sympy)
ENV VENV_PATH=/opt/compass_venv
RUN python3.10 -m venv $VENV_PATH \
  && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
  && ln -sf /usr/bin/python3.10 /usr/bin/python
ENV PATH="$VENV_PATH/bin:$PATH"

# Install COMPASS Python dependencies in venv
# For GPU support: docker build --build-arg PIP_EXTRA=https://download.pytorch.org/whl/cu121
ARG PIP_EXTRA=
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
  && if [ -n "$PIP_EXTRA" ]; then \
      pip install --no-cache-dir --extra-index-url "$PIP_EXTRA" -r /tmp/requirements-docker.txt; \
    else \
      pip install --no-cache-dir -r /tmp/requirements-docker.txt; \
    fi \
  && rm /tmp/requirements-docker.txt

# Working directory
WORKDIR /workspace/COMPASS

# Default: run training driver (uses venv python)
CMD ["python", "driver.py"]
