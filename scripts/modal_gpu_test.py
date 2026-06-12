"""Validate kernelmeter on a real NVIDIA GPU using Modal.

Run from the repo root:  modal run scripts/modal_gpu_test.py

Exercises the three claims the README makes: `info` works against a live
driver, `info --json` round-trips, and `bench` produces sane speed-of-light
numbers for the example kernels.
"""

import modal

app = modal.App("kernelmeter-gpu-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch")  # pulls in triton on linux
    .add_local_dir("src/kernelmeter", remote_path="/root/kernelmeter")
    .add_local_dir("examples", remote_path="/root/examples")
)


@app.function(gpu="T4", image=image, timeout=900)
def validate() -> dict:
    import contextlib
    import io
    import json

    from kernelmeter import bench as kb
    from kernelmeter import cli

    report = {}

    print("=== kernelmeter info ===")
    report["info_rc"] = cli.main(["info"])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report["info_json_rc"] = cli.main(["info", "--json"])
    payload = json.loads(buf.getvalue())
    dev = payload["devices"][0]
    report["device"] = dev["name"]
    report["attribute_count"] = len(dev["attributes"])
    report["unknown_attr_ids"] = sorted(
        k for k in dev["attributes"] if k.startswith("attribute_")
    )
    report["derived"] = dev["derived"]

    for example in (
        "/root/examples/vector_add.py",
        "/root/examples/softmax.py",
        "/root/examples/matmul.py",
    ):
        print(f"\n=== kernelmeter bench {example} ===")
        report[f"bench_rc:{example.rsplit('/', 1)[-1]}"] = cli.main(["bench", example])
        kb.REGISTRY.clear()  # registry is process-global; reset between files

    print("\n=== kernelmeter roofline ===")
    report["roofline_rc"] = cli.main(["roofline", "--ai", "0.33"])

    print("\n=== kernelmeter occupancy (live device) ===")
    report["occupancy_rc"] = cli.main(["occupancy", "--block", "256", "--regs", "40"])

    print("\n=== kernelmeter ceiling ===")
    report["ceiling_rc"] = cli.main(["ceiling", "--mb", "128", "--matmul-n", "2048"])

    return report


@app.local_entrypoint()
def main():
    report = validate.remote()
    print("\n=== summary (returned to local) ===")
    for key, value in report.items():
        print(f"{key}: {value}")
