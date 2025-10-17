
#!/bin/bash

# Ensure CUDA version is passed as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <cuda_version>"
    exit 1
fi

CUDA_VERSION=$1
echo "Setting up environment with CUDA version: $CUDA_VERSION"

# Install PyTorch and torchvision according to the specified CUDA version
case $CUDA_VERSION in
    10.2)
        pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu102/torch_stable.html
        ;;
    11.1)
        pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
        ;;
    12.1)
        pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu121/torch_stable.html
        ;;
    *)
        echo "CUDA version $CUDA_VERSION is not supported. Exiting."
        exit 2
        ;;
esac

# Install general dependencies
pip install ipywidgets==8.0.2 jupyterlab==3.4.2 lovely-tensors==0.1.15

# Navigate to the Dust3r directory and install its dependencies
cd dust3r
pip install -r requirements.txt
pip install -r requirements_optional.txt

# Compile RoPE positional embeddings
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../

# Download pre-trained model
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/

# Setup Gaussian splatting dependencies
pip install -r ../requirements.txt
pip install -e ../gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -e ../gaussian-splatting/submodules/simple-knn

cd ..

echo "Environment setup is complete."
