from flask import Flask, request, jsonify
from src.models.model import Model
import traceback
from termcolor import cprint

app = Flask(__name__)

classifier = Model()
model = classifier.load(path='./models/news_classifier.joblib')    
cprint ('Model loaded', 'red')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if model:
        try:
            # Get the data from the POST request.
            data = request.get_json(force=True)
            print("data")
            pred = model.predict(data['title'])
            return jsonify({'prediction': pred})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        cprint('Train the model first', 'red')
        return 'No model to use'

if __name__ == "__main__":
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 12345 # If you don't provide any port then the port will be set to 12345
    app.run(port=port, debug=True)