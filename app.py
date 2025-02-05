from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("naive_bayes_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)