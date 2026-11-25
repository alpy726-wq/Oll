import sys; sys.path.insert(0, "gpu"); import modal_env  # noqa
import modal
app = modal.App("oll-probe")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.9.0", "numpy")

@app.function(gpu="B200", image=image, timeout=600)
def probe():
    import torch, subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout
    x = torch.randn(8192, 8192, device="cuda"); torch.cuda.synchronize()
    import time; t = time.time()
    for _ in range(20): y = x @ x
    torch.cuda.synchronize()
    return dict(gpu=out.strip(), torch=torch.__version__, tflops=round(20 * 2 * 8192 ** 3 / (time.time() - t) / 1e12, 1))

if __name__ == "__main__":
    with modal.enable_output(), app.run():
        print("RESULT", probe.remote())
