import pytest
import pandas as pd
from sklearn.exceptions import NotFittedError
from src.models.model import Model

@pytest.fixture
def input_model():
    return Model()

def test_load_data(input_model):
    with pytest.raises(FileNotFoundError):
        input_model._load_data('foo.bar')

def test_predict(input_model):
    with pytest.raises(NotFittedError):
        input_model.predict('Foo')

    input_model.train('data/raw/newsCorpora.csv')
    pred = input_model.predict('Foo')
    assert isinstance(pred, str)
    assert pred in input_model.label_names.values()

def test_eval(input_model):
    with pytest.raises(FileNotFoundError):
        input_model.eval('Foo')
    
    with pytest.raises(NotFittedError):
        input_model.eval('data/raw/newsCorpora.csv')

    input_model.train('data/raw/newsCorpora.csv')
    accuracy = input_model.eval('data/raw/newsCorpora.csv')
    assert isinstance(accuracy, float)