"""Fail-fast verification for the repository's PyTorch/CUDA environment."""

import json

import torch


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this virtual environment")
    capability = torch.cuda.get_device_capability(0)
    left = torch.randn(1024, 1024, device="cuda")
    right = left @ left
    torch.cuda.synchronize()
    result = {
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(capability),
        "compiled_arches": torch.cuda.get_arch_list(),
        "matmul_finite": bool(torch.isfinite(right).all()),
    }
    if capability >= (12, 0) and "sm_120" not in result["compiled_arches"]:
        raise SystemExit(f"Installed wheel does not contain Blackwell sm_120: {result}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

