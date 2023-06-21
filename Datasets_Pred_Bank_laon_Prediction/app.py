import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    prediction_text = " "
    if output == 1:
        prediction_text = "Your Credit is Approved  "
    if output == 0:
        prediction_text = "Your Credit is Refused "

    return render_template('index.html', prediction_text=prediction_text.format(output))

if __name__ == "__main__":
    app.run(debug=True)