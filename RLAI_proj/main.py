import argparse
from train import train_model

def main():
    parser = argparse.ArgumentParser(description="Train an AI model.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of training iterations.")
    args = parser.parse_args()

    for _ in range(args.iterations):
        train_model()

if __name__ == "__main__":
    main()