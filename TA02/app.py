import os
from flask import Flask, flash, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask import session
import pandas as pd
import csv


UPLOAD_PATHS = "/home/amzesmoro/Documents/Semester 7/TA/implementasi/implement_to_flask/FinalProjectTA-1920-02/TA02/static/upload_files"

ALLOWED_EXTENSIONS ={ 'xlsx', 'xls', 'csv'}

app = Flask(__name__)
app.config["UPLOAD_PATHS"] = UPLOAD_PATHS

def allowed_file(filename):
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

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_PATHS"], filename))
            #return redirect(url_for("uploaded_file", filename=filename))            
            return redirect(request.url)
    return render_template("home.html")

# @app.route("/", methods=["GET", "POST"])    
# def home():
#     if request.method == "POST":
#         f = request.form["file"]
#         data = []
#         with open(f) as file:
#             csvfile = csv.reader(file)
#             for row in csvfile:
#                 data.append(row)
#         data = pd.DataFrame(data)                
#         return render_template("home.html", shape=data.shape, data=data.to_html())
#     return render_template("home.html")

@app.route("/test")
def test():
    df = pd.read_csv("/home/amzesmoro/Documents/Semester 7/TA/implementasi/dataset_sample.csv")
    return df.to_html()

@app.route("/uploads/<filename>")        
def uploaded_file(filename):
    #return send_from_directory(app.config["UPLOAD_PATHS"], filename)
    return redirect(url_for("home"))

@app.route("/holt-winters")
def holt_winters():
    return render_template("holt-winters.html")

@app.route("/holt-winters-2", methods=["POST"])
def holt_winters_sts():
    if request.method == "POST":
        req = request.form
        alpha = req["alpha"]
        beta = req["beta"]
        gamma = req["gamma"]
        return render_template("holt-winters-2.html", alpha = alpha, beta = beta, gamma = gamma)
  
@app.route("/predict-holt-winters")
def predict_holt_winters():
    return render_template("predict-holt-winters.html")

@app.route("/svr-preprocessing")
def svr_preprocessing():
    return render_template("svr_preprocessing.html")

@app.route("/svr-uni", methods=["POST"])
def svr_univariate():
    if request.method == "POST":
        req = request.form
        trainUni = req["train-uni"]
        testUni = req["test-uni"]
        return render_template("svr-uni.html", trainUni = trainUni, testUni = testUni)
    return render_template("svr-uni.html")

@app.route("/svr-multi", methods=["POST"])
def svr_multivariate():
    if request.method == "POST":
        req = request.form
        trainMulti = req["train-multi"]
        testMulti = req["test-multi"]
        return render_template("svr-multi.html", trainMulti = trainMulti, testMulti = testMulti)
    return render_template("svr-multi.html")

@app.route("/predict-svr-uni", methods=["POST"])
def predict_svr_univariate():
    if request.method == "POST":
        req = request.form
        clrUni = req["clr-uni"]
        cUni = req["c-uni"]
        epsilonUni = req["epsilon-uni"]
        lambdaUni = req["lambda-uni"]
        sigmaUni = req["sigma-uni"]
        iterationUni = req["iteration-uni"]        
        return render_template("predict-svr-uni.html",
        clrUni = clrUni, cUni = cUni, epsilonUni = epsilonUni, 
        lambdaUni = lambdaUni, sigmaUni = sigmaUni, iterationUni = iterationUni)
    return render_template("predict-svr-uni.html")

@app.route("/predict-svr-multi", methods=["POST"])
def predict_svr_multivariate():
    if request.method == "POST":
        req = request.form
        clrMulti = req["clr-multi"]
        cMulti = req["c-multi"]
        epsilonMulti = req["epsilon-multi"]
        lambdaMulti = req["lambda-multi"]
        sigmaMulti = req["sigma-multi"]
        iterationMulti = req["iteration-multi"]        
        return render_template("predict-svr-multi.html",
        clrMulti = clrMulti, cMulti = cMulti, epsilonMulti = epsilonMulti, 
        lambdaMulti = lambdaMulti, sigmaMulti = sigmaMulti, iterationMulti = iterationMulti)
    return render_template("predict-svr-multi.html")

@app.route("/about/")
def about():
    return render_template("about.html")


if __name__ == "__main___":
    app.run(debug=True)