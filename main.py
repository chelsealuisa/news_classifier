import argparse
import sys
from termcolor import cprint
from src.models.model import Model
from typing import List, Any
import logging

def create_cmd_parser():
    parser = argparse.ArgumentParser(description='News headlines classifier.')
    parser.add_argument('--log_level', '-l', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='DEBUG')
    subparsers = parser.add_subparsers(help='mode of operation', dest='cmd')
    subparsers.required = True

    train_parser = subparsers.add_parser('train', help='train a model')
    train_parser.set_defaults(cmd='train')
    train_parser.add_argument('input_path', help='data to train on')
    train_parser.add_argument('--params', '-p', default='None', help='parameters to train model with')
    train_parser.add_argument('--cv', '-cv', type=int, default='0', help='number of folds to use for cross validation')
    train_parser.add_argument('--modelpath', '-m', default='./models/', help='path to folder in which to save the model')

    predict_parser = subparsers.add_parser('predict', help='predict a single example')
    predict_parser.set_defaults(cmd='predict')
    predict_parser.add_argument('example', help='example headline to predict')
    predict_parser.add_argument('--modelpath', '-m', default='./models/news_classifier.joblib', help='path to model file to load')

    predict_parser = subparsers.add_parser('evaluate', help='evaluate a trained model')
    predict_parser.set_defaults(cmd='evaluate')
    predict_parser.add_argument('test_path', help='data to evaluate on')
    predict_parser.add_argument('--modelpath', '-m', default='./models/news_classifier.joblib', help='path to model file to load')
    return parser

def run(args: List[Any] = None) -> None:
    parser = create_cmd_parser()
    args = parser.parse_args(args=args)
    classifier = Model()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=format('%(asctime)s : %(levelname)s : %(name)s : [%(funcName)s] : %(message)s'),
        filename=('./log/newsclassifier.log')
    )

    if args.cmd == 'train':
        classifier.train(input_path=args.input_path, params=args.params, cv=args.cv)
        classifier.save(path=args.modelpath)

    if args.cmd == 'predict':
        model = classifier.load(path=args.modelpath)
        prediction = model.predict(example=args.example)
        cprint(prediction, 'green')

    if args.cmd == 'evaluate':
        model = classifier.load(path=args.modelpath)
        score = model.eval(test_data=args.test_path)
        cprint(f'Model accuracy: {score}', 'green')


if __name__=='__main__':
    run(sys.argv[1:])