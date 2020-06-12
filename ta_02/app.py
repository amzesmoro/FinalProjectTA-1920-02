import os
from flask import Flask, flash, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import csv


UPLOAD_PATHS = "/home/amzesmoro/Documents/Semester 7/TA/implementasi/implement_to_flask/ta_02/static/upload_files"
ALLOWED_EXTENSIONS ={ 'xlsx', 'xls', 'csv'}

app = Flask(__name__)
app.config["UPLOAD_PATHS"] = UPLOAD_PATHS

def allowedFile(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])    
def home():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"] 

        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)

        if file and allowedFile(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_PATHS"], filename))
            #return redirect(url_for("uploadedFile", filename=filename))
            return redirect(request.url)    
    return render_template("home.html")

@app.route("/uploads/<filename>")        
def uploadedFile(filename):
    #return send_from_directory(app.config["UPLOAD_PATHS"], filename)
    return redirect(url_for("home"))

@app.route("/holt-winters")
def holtWinters():
    return render_template("holt-winters.html")

@app.route("/holt-winters-2")
def holtWintersSts():
    return render_template("holt-winters-2.html")

@app.route("/predict-holt-winters")
def predictHoltWinters():
    return render_template("predict-holt-winters.html")

@app.route("/svr", methods=["GET", "POST"])
def svr():
    if request.method == "GET":
        return render_template("svr.html")
    elif request.method == "POST":
        trainUni = request.form["train-uni"]
        testUni = request.form["test-uni"]
        trainMulti = request.form["train-multi"]
        testMulti = request.form["test-multi"]
        return render_template("svr-uni.html")

@app.route("/svr-uni")
def svrUnivariate():
    return render_template("svr-uni.html")

@app.route("/svr-multi")
def svrMultivariate():
    return render_template("svr-multi.html")

@app.route("/predict-svr-uni")
def predictSvrUnivariate():
    return render_template("predict-svr-uni.html")

@app.route("/predict-svr-multi")
def predictSvrMultivariate():
    return render_template("predict-svr-multi.html")

@app.route("/about/")
def about():
    return render_template("about.html")


if __name__ == "__main___":
    app.run(debug=True)