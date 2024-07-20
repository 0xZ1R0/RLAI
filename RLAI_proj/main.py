import argparse
from train import train_model
from visualize import visualize_model

def main():
    parser = argparse.ArgumentParser(description="Train and visualize an AI model.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the AI in 3D.")
    args = parser.parse_args()

    model = train_model()

    if args.visualize:
        visualize_model(model)
    else:
        print("Model training complete. Use --visualize to see the AI in 3D.")

if __name__ == "__main__":
    main()