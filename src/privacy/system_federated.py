import subprocess


def run_federated_from_privacy(config_path="configs/federated.yaml"):
    """
    Bridge function: allow privacy pipeline to trigger federated experiment.
    """
    cmd = ["python", "src/run_federated.py", "--config", config_path]
    print(f"[INFO] Launch federated simulation: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
