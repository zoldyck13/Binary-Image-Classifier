#!/bin/bash
set -e

echo "Detecting package manager..."

if command -v pacman &> /dev/null; then
    echo "Arch Linux detected (pacman)"
    sudo pacman -Syu --needed --noconfirm base-devel cmake cuda opencv
elif command -v apt &> /dev/null; then
    echo "Debian/Ubuntu detected (apt)"
    sudo apt update
    sudo apt install -y build-essential cmake pkg-config \
                        libopencv-dev nvidia-cuda-toolkit
else
    echo "Unsupported package manager. Please install CUDA + OpenCV manually."
    exit 1
fi

echo "Dependencies installed."

# Verify nvcc
if ! command -v nvcc &> /dev/null; then
    echo "nvcc not found. Make sure CUDA bin directory is in your PATH."
    echo "Try: echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc"
else
    echo "nvcc found: $(nvcc --version | head -n 1)"
fi

# Verify OpenCV
if pkg-config --modversion opencv4 &> /dev/null; then
    echo "OpenCV version: $(pkg-config --modversion opencv4)"
elif pkg-config --modversion opencv &> /dev/null; then
    echo "OpenCV version: $(pkg-config --modversion opencv)"
else
    echo "OpenCV not found by pkg-config, but it might still be installed."
fi
