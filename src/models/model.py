import pandas as pd
import joblib

class Model():

    def __init__(self, model = None, parameters = None, title = 'news_classifier', label_col = 'CATEGORY', text_col = 'TITLE'):
        
        if isinstance(model, str):
            self.load(model)
        else:
            self.model = model

        self.parameters = None
        self.label_column = label_col
        self.text_column = text_col

 
    def _load_data(self, input_path):
        """
        Loads data from csv file given by ``input_path``.

        Args:
            input_path: path to the input csv.

        """
        if not input_path.endswith('.csv'):
            cprint('Model cannot be loaded. The path to the input file has to refer to a .csv file', 'red')
            raise FileNotFoundError('Input file path has to end in .csv.')

        headers = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
        df = pd.read_csv(input_path, sep='\t', header=None, names=headers)
        df = df.sample(10000)
        
        X = df[self.text_column]
        y = df[self.label_column]

        return X, y
    
 
    def train(self, input_path: str, params: Dict = None, cv: Int = 5) -> None:
        '''
        Trains a classification model on the given dataset contained in ``input_path``.

        Args:
            input_path: path to the input csv.
            params: parameters to train model with when not using cv; if given, ``self.params`` is not used.
            cv: number of folds to use for cross validation; if None, cross validation is not performed.
            
        '''
        X, y = self._load_data(input_path)

        pipe = Pipeline(steps=[
            ('tfidf', TfidfVectorizer()),
            ('clf', SGDClassifier(random_state=100))
        ])

        if cv not None:
            param_grid = {
                'tfidf__ngram_range': [(1,1), (1,2)],
                'clf__loss': ['log', 'hinge', 'modified_huber'],
                'clf__penalty': ['l1','l2','elasticnet'], 
                'clf__alpha': [1e-3, 1e-2, 1e-1, 1e0],
                'clf__shuffle': [True],
                'clf__max_iter': [500, 1000, 2000],
            }
            search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=cv, scoring='accuracy')
            search.fit(X, y)
            self.model = search.best_estimator_

        elif params is None:
            params = self.params

        self.model = pipe.fit(X, y)


    def predict(self, examples):
        pass


    def save(self, path):
        filepath = os.path.join(path, f'{self.title}_{get_timestamp_str()}.joblib')
        cprint(f'Saving the model to : {filepath}', 'green')
        joblib.dump(self, filepath)


    @staticmethod
    def load(path):
        if not path.endswith('.joblib'):
            cprint('Model cannot be loaded. The path to the model has to refer to .joblib file', 'red')
            raise FileNotFoundError('Model path has to end with .joblib')
        return joblib.load(path)


    def run(self, args):
        pass


if __name__ == '__main__':
    model = Model()
    model.run()