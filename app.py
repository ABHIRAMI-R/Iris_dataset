from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Getting data from HTML form
    sep_len = float(request.form["sepal_length"])
    sep_wid = float(request.form["sepal_width"])
    pet_len = float(request.form["petal_length"])
    pet_wid = float(request.form["petal_width"])

    # Convert to array format
    features = np.array([[sep_len, sep_wid, pet_len, pet_wid]])

    # Prediction
    pred = model.predict(features)[0]

    species = ["Setosa", "Versicolor", "Virginica"][pred]

    return render_template("result.html", prediction=species)

if __name__ == "__main__":
    app.run(debug=True)