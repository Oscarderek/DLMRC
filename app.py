from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

app = Flask(__name__)

reg_model = tf.keras.models.load_model("stay_duration_model.h5", compile=False)
clf_model = tf.keras.models.load_model("care_level_model.h5", compile = False)

preprocessor = joblib.load("preprocessor.pkl")

# Label decoder for classification output
label_decoder = {0: "Emergency", 1: "General", 2: "HDU", 3: "ICU"}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_stay_duration", methods=["GET", "POST"])
def predict_stay_duration():
    if request.method == "POST":
        input_data = get_input_data(request)
        transformed = preprocessor.transform(pd.DataFrame([input_data]))
        prediction = reg_model.predict(transformed)[0][0]
        return render_template("predict_stay_duration.html", prediction=round(prediction, 2))
    return render_template("predict_stay_duration.html")

@app.route("/predict_care_level", methods=["GET", "POST"])
def predict_care_level():
    if request.method == "POST":
        input_data = get_input_data(request)
        transformed = preprocessor.transform(pd.DataFrame([input_data]))
        prediction = clf_model.predict(transformed)
        label_index = np.argmax(prediction)
        care_level = label_decoder[label_index]
        return render_template("predict_care_level.html", prediction=care_level)
    return render_template("predict_care_level.html")

def get_input_data(req):
    return {
        "age": int(req.form["age"]),
        "gender": req.form["gender"],
        "pre_condition": req.form["pre_condition"],
        "admission_reason": req.form["admission_reason"],
        "bp_systolic": int(req.form["bp_systolic"]),
        "bp_diastolic": int(req.form["bp_diastolic"]),
        "pulse": int(req.form["pulse"]),
        "oxygen": float(req.form["oxygen"]),
        "wbc_count": float(req.form["wbc_count"]),
        "creatinine": float(req.form["creatinine"]),
        "comorbidity_index": int(req.form["comorbidity_index"]),
        "admission_time": req.form["admission_time"],
        "department": req.form["department"],
        "emergency_status": int(req.form["emergency_status"]),
        "insurance": req.form["insurance"]
    }

if __name__ == "__main__":
    app.run(debug=True)