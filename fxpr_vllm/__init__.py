"""Fixed point reductions for VLLM."""

# Importing library_ops at package load time guarantees that:
#   1. The CUDA extension (`_cuda`) is loaded and `torch.ops.fxpr.*` is
#      registered.
#   2. Every op also has a fake/meta implementation registered, so
#      torch.compile / dynamo tracers don't try to run the CUDA kernel
#      against FakeTensors.
# Skipping this import would cause "tensor has a non-zero number of
# elements, but its data is not allocated yet" under torch.compile.
from . import library_ops  # noqa: F401
