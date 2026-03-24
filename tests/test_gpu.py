import time
import ctypes.util
import torch
import pyopencl as cl


def check_opencl_runtime():
    """Check if OpenCL ICD loader and vendor runtimes are available."""
    # Step 1: check for loader
    loader = ctypes.util.find_library("OpenCL")
    if loader is None:
        raise RuntimeError(
            "OpenCL ICD loader (libOpenCL.so) not found.\n"
            "👉 Install it on Ubuntu with:\n"
            "   sudo apt install ocl-icd-opencl-dev clinfo\n"
            "Or install `ocl-icd-system` in your Pixi/conda env."
        )

    # Step 2: check for platforms
    try:
        platforms = cl.get_platforms()
    except cl._cl.LogicError as e:
        raise RuntimeError(
            "OpenCL loader found, but no vendor ICD runtime detected.\n"
            "For NVIDIA GPUs, please install the NVIDIA driver with OpenCL support "
            "(e.g. `libnvidia-compute-<version>` on Ubuntu).\n"
            "👉 To fix inside Pixi, run:\n"
            "   pixi run fix_opencl"
        ) from e

    if not platforms:
        raise RuntimeError(
            "No OpenCL platforms detected. "
            "This usually means the NVIDIA OpenCL ICD (libnvidia-opencl.so) is missing."
        )

    return platforms


def run_pytorch_gpu_test():
    """Run a simple GPU test with PyTorch to verify CUDA/GPU access."""
    print("\nPyTorch version:", torch.__version__)
    cuda_available = torch.cuda.is_available()

    if not cuda_available:
        print("❌ No GPU detected by PyTorch (torch.cuda.is_available() is False).")
        return

    device_count = torch.cuda.device_count()
    print(f"\n✅ {device_count} GPU(s) detected:")
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        print(f"  ID {i}: {name} (total memory: {props.total_memory / 2**30:.2f} GB)")

    print("\nRunning GPU test on cuda:0 ...")
    device = torch.device("cuda:0")
    for i in range(5):
        a = torch.randn(10000, 10000, device=device)
        b = torch.randn(10000, 10000, device=device)
        c = torch.matmul(a, b)
        print(f"Iteration {i+1}: result shape {c.shape}, device: {c.device}")
        time.sleep(1)  # pause so you can see activity in nvidia-smi


if __name__ == "__main__":
    # Step 1: Check OpenCL
    try:
        plats = check_opencl_runtime()
        print("✅ OpenCL platforms available:")
        for p in plats:
            print(" -", p.name)
    except RuntimeError as err:
        print("❌", err)

    # Step 2: Run PyTorch GPU test
    run_pytorch_gpu_test()

