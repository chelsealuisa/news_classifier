import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from typing import Dict

class Model():

    def __init__(self, model = None, parameters = None, title = 'news_classifier', label_col = 'CATEGORY', text_col = 'TITLE'):
        
        if isinstance(model, str):
            self.load(model)
        else:
            self.model = model

        self.parameters = None
        self.label_column = label_col
        self.text_column = text_col
        self.labels_names = {
            'b': 'Business',
            't': 'Science and Technology',
            'e': 'Entertainment',
            'm': 'Health'
        }

 
    def _load_data(self, input_path):
        """
        Loads data from csv file given by ``input_path``.

        Args:
            input_path: path to the input csv.
        Returns:
            X: feature dataframe.
            y: target variable dataframe.

        """
        if not input_path.endswith('.csv'):
            print('Model cannot be loaded. The path to the input file has to refer to a .csv file', 'red')
            raise FileNotFoundError('Input file path has to end in .csv.')

        headers = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
        df = pd.read_csv(input_path, sep='\t', header=None, names=headers)
        df = df.sample(10000)
        
        X = df[self.text_column]
        y = df[self.label_column]

        return X, y
    
 
    def train(self, input_path: str, params: Dict = None, cv: int = None) -> None:
        '''
        Trains a classification model on the given dataset contained in ``input_path``.

        Args:
            input_path: path to the input csv.
            params: parameters to train model with when not using cv; if given, ``self.params`` is not used.
            cv: number of folds to use for cross validation; if None, cross validation is not performed.
            
        '''
        X, y = self._load_data(input_path)

        if cv is not None:
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
            
            search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=cv, scoring='accuracy')
            self.model = search.fit(X, y)
            return 

        elif params is None:
            params = self.params

        pipe = Pipeline(steps=[
                ('tfidf', TfidfVectorizer()),
                ('clf', SGDClassifier(**params, random_state=100))
        ])
        self.model = pipe.fit(X, y)
        return


    def predict(self, example: str):
        """
        Returns model prediction for a single example.

        Args:
            example: sample news headline to be predicted, as a string.
        Return:
            string: the predicted label.
        """
        y_pred = self.model.predict([example])[0]
        label = self.labels_names[y_pred]
        return label

    def save(self, path):
        filepath = os.path.join(path, f'{self.title}_{get_timestamp_str()}.joblib')
        cprint(f'Saving the model to : {filepath}', 'green')
        joblib.dump(self, filepath)
        return

    @staticmethod
    def load(path):
        if not path.endswith('.joblib'):
            cprint('Model cannot be loaded. The path to the model has to refer to .joblib file', 'red')
            raise FileNotFoundError('Model path has to end with .joblib')
        return joblib.load(path)


    def run(self, args=None):
        pass


if __name__ == '__main__':
    model = Model()
    model.train("data/raw/newsCorpora.csv", cv=2)
    label = model.predict("Man landed on the moon")
    print(label)