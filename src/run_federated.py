import argparse
from federated.fed_loop import run_federated_training


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/privacy_federated.yaml",
        help="Path to federated config yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_federated_training(args.config)
