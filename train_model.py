import argparse
from train_model.training.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--groundtruth", required=True)
    parser.add_argument("--modelpath", required=True)
    args = parser.parse_args()

    train(args.rgb, args.groundtruth, args.modelpath)
