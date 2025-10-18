import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device count:", torch.cuda.device_count())
    print("Current GPU device:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Memory allocated (MB):", round(torch.cuda.memory_allocated(0) / 1024**2, 2))
    print("Memory reserved (MB):", round(torch.cuda.memory_reserved(0) / 1024**2, 2))

    # Optional: Run a quick tensor test
    x = torch.rand(10000, 10000, device='cuda')
    print("Tensor successfully created on GPU ✅")
else:
    print("No GPU detected ❌ — check your CUDA/PyTorch installation.")
