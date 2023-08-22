import pickle
from flask import Flask, render_template, redirect, request, jsonify ,url_for
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from urllib.request import urlopen

app = Flask(__name__)

## import ridge model and stadndard scaler pickel file
linear_model = pickle.load(open("models/linear_regression.pkl","rb"))
Standard_scaler = pickle.load(open("models/scaler.pkl","rb"))

## route from home page

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/my_data", methods = ['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        TOEFL_Score = float(request.form.get("TOEFL_Score"))
        CGPA = float(request.form.get("CGPA"))
        SOP = float(request.form.get("SOP"))
        LOR = float(request.form.get("LOR"))
        University_Rating = float(request.form.get("University_Rating"))
        GRE_Score = float(request.form.get("GRE_Score"))
        Research = float(request.form.get("Research"))

        new_data = Standard_scaler.transform([[TOEFL_Score,CGPA,SOP,LOR,University_Rating,GRE_Score,Research]])
        resultt = linear_model.predict(new_data)

        return render_template("index.html",result=resultt[0]*100)

    else :  
        return render_template("index.html")


if __name__=="__main__":
    app.run(host="0.0.0.0" ,debug=True)
