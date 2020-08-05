import pandas as pd
import joblib
import os
from termcolor import cprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score
from sklearn.exceptions import NotFittedError
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from IPython import embed
import logging

class Model():
    """
    Creates a text classification model.

    Args:
        model: path to a model to load.
        parameters: model parameters to use. 
        title: title of the model.
        label_col: column name of the target variable.
        text_col: column name of the text to be calssified.

    """
    def __init__(self, model = None, parameters = None, title = 'news_classifier', 
    label_col = 'CATEGORY', text_col = 'TITLE') -> None:
        self.logger = logging.getLogger(__name__)

        if isinstance(model, str):
            self.model = self.load(model)
        else:
            self.model = model
        self.title = title
        self.parameters = parameters
        self.label_column = label_col
        self.text_column = text_col
        self.labels_names = {
            'b': 'Business',
            't': 'Science and Technology',
            'e': 'Entertainment',
            'm': 'Health'
        }

 
    def _load_data(self, input_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads data from csv file given by ``input_path``.

        Args:
            input_path: path to the input csv.
        Returns:
            X: feature dataframe.
            y: target variable dataframe.

        """
        if not input_path.endswith('.csv'):
            self.logger.error('Model cannot be loaded. The path to the input file has to refer to a .csv file')
            raise FileNotFoundError('Input file path has to end in .csv.')

        self.logger.info(f'Loading data from file: {input_path}.')

        headers = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
        df = pd.read_csv(input_path, sep='\t', header=None, names=headers)
        df = df.sample(10000)
        
        X = df[self.text_column]
        y = df[self.label_column]

        self.logger.info(f'Loaded data. Data size: {df.shape}.')

        return X, y
    
 
    def train(self, input_path: str, params: Dict = None, cv: int = 0) -> None:
        """
        Trains a classification model on the given dataset contained in ``input_path``.

        Args:
            input_path: path to the input csv.
            params: parameters to train model with when not using cv; if given, ``self.params`` is not used.
            cv: number of folds to use for cross validation; if None, cross validation is not performed.
            
        """
        X, y = self._load_data(input_path)

        if cv > 0:
            param_grid = {
                'tfidf__ngram_range': [(1,1), (1,2)],
                'clf__loss': ['log', 'hinge', 'modified_huber'],
                'clf__penalty': ['l1','l2','elasticnet'], 
                'clf__alpha': [1e-3, 1e-2, 1e-1, 1e0],
                'clf__shuffle': [True],
                'clf__max_iter': [500, 1000, 2000],
            }
            
            pipe = Pipeline(steps=[
                ('tfidf', TfidfVectorizer()),
                ('clf', SGDClassifier(random_state=100))
            ])
            self.logger.info(f'Training model on {input_path} with {cv}-fold grid search CV.')
            self.model = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=cv, scoring='accuracy')
            self.model.fit(X, y)
            ## TODO: joblib save and load don't keep params stored
            self.parameters = self.model.best_params_
            return 

        elif params is not None:
            self.params = params
        
        self.logger.info(f'Training model on {input_path} without grid search CV. Parameters: {self.params}.')

        self.model = Pipeline(steps=[
                ('tfidf', TfidfVectorizer()),
                ('clf', SGDClassifier(**self.params, random_state=100))
        ])
        self.model.fit(X, y)
        return


    def predict(self, example: str) -> str:
        """
        Returns model prediction for a single example.

        Args:
            example: sample news headline to be predicted, as a string.
        Return:
            string: the predicted label.

        """
        if self.model is None:
            self.logger.error('Model not fitted before evaluation.')
            raise NotFittedError('Trained model not found. A model needs to be trained before prediction.')

        y_pred = self.model.predict([example])[0]
        label = self.labels_names[y_pred]
        return label


    def eval(self, test_data: str) -> float:
        """
        Evaluate performance of trained model. Saves a .csv containing the classification report
        and .png file contaning a plot of the confusion matrix.

        Args:
            test_data: path to test dataset.

        Return:
            accuracy of the trained model.

        """
        X_test, y_test = self._load_data(test_data)

        if self.model is None:
            self.logger.error('Model not fitted before evaluation.')
            raise NotFittedError('Trained model not found. A model needs to be trained before evaluation.')

        y_pred = self.model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        pd.DataFrame(report).transpose().to_csv('reports/classification_report.csv')
        
        plot_confusion_matrix(self.model, X_test, y_test,
                                 display_labels=y_test.unique(),
                                 cmap=plt.cm.Blues,
                                 normalize=None);
        plt.savefig('reports/confusion_matrix.png');

        self.logger.info(f'Evaluating model. Model accuracy: {accuracy_score(y_test, y_pred)}')

        return accuracy_score(y_test, y_pred)

    def save(self, path):
        filepath = os.path.join(path, f'{self.title}.joblib')
        self.logger.info(f'Saving model to: {filepath}.')
        joblib.dump(self, filepath)
        return

    @staticmethod
    def load(path):
        if not path.endswith('.joblib'):
            self.logger.error('Model cannot be loaded. The path to the model has to refer to .joblib file.')
            raise FileNotFoundError('Model path has to end with .joblib')
        return joblib.load(path)


if __name__ == '__main__':
    model = Model()
    cprint('Training model', 'green')
    params = {
        'alpha': 0.001, 
        'loss': 'modified_huber', 
        'max_iter': 500, 
        'penalty': 'l2',
        'shuffle': True, 
        }
    model.train("data/raw/newsCorpora.csv", params=params)
    cprint('Model trained', 'green')
    
    cprint('Save model', 'green')
    model.save("./models/")
    
    cprint('Load model', 'green')
    cprint('Model params:', 'green')
    reloaded_model = Model.load("./models/news_classifier.joblib")
    cprint(reloaded_model.parameters, 'green')
    
    cprint('Prediction for "Man landed on the moon":', 'green')
    label = model.predict('Man landed on the moon')
    cprint(label, 'green')
    
    cprint('Evaluate model', 'green')
    cprint(model.eval('data/raw/newsCorpora.csv'), 'green')
    