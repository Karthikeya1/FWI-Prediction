from flask import Flask,jsonify,render_template,request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application=Flask(__name__)
app=application

##import ridge regressor and standard scalar

ridge_model=pickle.load(open("models/ridge.pkl","rb"))
standard_scalar=pickle.load(open("models/scale.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method=="POST":
        Temperature=float(request.form.get("Temperature"))
        RH=float(request.form.get("RH"))
        Ws=float(request.form.get("Ws"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        Classes=float(request.form.get("Classes"))
        Region=float(request.form.get("Region"))

        scaled_data=standard_scalar.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(scaled_data)
        return render_template("home.html",results=result[0])
    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)