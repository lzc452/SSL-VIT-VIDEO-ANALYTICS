import subprocess


def run_federated_from_privacy():
    """
    Bridge function so privacy pipeline can trigger federated experiment.
    """
    cmd = ["python", "src/run_federated.py"]
    print("[INFO] Launch federated simulation from system_federated")
    subprocess.run(cmd, check=True)
