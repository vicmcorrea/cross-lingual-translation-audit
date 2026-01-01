"""Fail early when the container does not have a usable CUDA device."""

import json

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")

payload = {
    "cuda_available": True,
    "cuda_version": torch.version.cuda,
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
}
print(json.dumps(payload, sort_keys=True))
