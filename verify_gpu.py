import sys
import torch

print("Python exe:", sys.executable)
print("Torch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_availabe())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. PyTorch is not using the GPU.")


print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:", torch.cuda.get_device_name(i))
    print(f"GPU {i} capability:", torch.cuda.get_device_capability(i))

try:
    print("Arch list:", torch.cuda.get_arch_list())
except Exception as e:
    print("Could not get arch list:", e)

x = torch.randn(1, device="cuda")
print("CUDA tensor test:", x)
print("GPU verification passed.")