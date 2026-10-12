# ─── Cell 1: environment + GPU checks (hard stop on failure) ───
import platform, sys
import numpy, pandas, scipy, torch

n_gpus = torch.cuda.device_count()
assert n_gpus >= 1, "No CUDA GPU visible — set the Kaggle accelerator to GPU T4 x2."
gpu_info = []
for i in range(n_gpus):
    cc = torch.cuda.get_device_capability(i)
    name = torch.cuda.get_device_name(i)
    gpu_info.append({"index": i, "name": name, "capability": list(cc)})
    assert cc >= (7, 0), (f"GPU {i} ({name}) has compute capability {cc[0]}.{cc[1]} < 7.0 "
                          f"(P100 is sm_60) — select GPU T4 x2.")
print(f"{n_gpus} GPU(s):", [g["name"] for g in gpu_info])
if n_gpus == 1:
    print("Only 1 GPU detected: the notebook runs sequentially; the budget planner accounts for it.")

run_manifest = {
    "python": sys.version.split()[0], "platform": platform.platform(),
    "torch": torch.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__,
    "scipy": scipy.__version__, "gpus": gpu_info, "n_gpus": n_gpus,
    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "config_sha256": CONFIG_SHA256, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(RUN_T0)),
}
(WORK / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
