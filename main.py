import argparse
import sys
from termcolor import cprint
from src.models.model import Model
from typing import List, Any

def create_cmd_parser():
    parser = argparse.ArgumentParser(description="News headlines classifier.")
    subparsers = parser.add_subparsers(help="mode of operation", dest="cmd")
    subparsers.required = True

    train_parser = subparsers.add_parser("train", help="train a model")
    train_parser.set_defaults(cmd="train")
    train_parser.add_argument("input_path", help="data to train on")
    train_parser.add_argument("--params", "-p", default="None", help="parameters to train model with")
    train_parser.add_argument("--cv", '-cv', type=int, default="None", help="number of folds to use for cross validation")
    train_parser.add_argument("--modelpath", "-m", default="./models/", help="path to folder in which to save the model")

    predict_parser = subparsers.add_parser("predict", help="predict a data set")
    predict_parser.set_defaults(cmd="predict")
    predict_parser.add_argument("example", help="example headline to predict")
    predict_parser.add_argument("--modelpath", "-m", default="./models/news_classifier.joblib", help="path to model file to use")

    return parser

def run(args: List[Any] = None) -> None:
    parser = create_cmd_parser()
    args = parser.parse_args(args=args)
    classifier = Model()

    if args.cmd == "train":
        classifier.train(input_path=args.input_path, params=args.params, cv=args.cv)
        classifier.save(path=args.modelpath)

    if args.cmd == "predict":
        model = classifier.load(path=args.modelpath)
        prediction = model.predict(example=args.example)
        cprint(prediction, 'green')


if __name__=="__main__":
    run(sys.argv[1:])