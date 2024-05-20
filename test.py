import torch
if torch.cuda.is_available():
    print("PyTorch with GPU support is successfully installed!")
    print(torch.__version__)  # Verify PyTorch version
    print(torch.version.cuda) # Verify CUDA version
else:
    print("PyTorch is installed, but GPU support is not available.") 
