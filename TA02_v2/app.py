import os
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
)

import pandas as pd
import csv
import holtwintersfunctions as hwf
import svrfunctions as svrf

from sklearn.preprocessing import MinMaxScaler
import math

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


## ROUTE
@app.route("/holt-winters", methods=["GET", "POST"])
def holt_winters():
    if request.method == "POST":
        f = request.form["csv_file"]
        data_csv = []
        with open(f) as file:
            csvfile = csv.reader(file)
            for row in csvfile:
                data_csv.append(row)
        data_csv = pd.DataFrame(data_csv)
        return render_template(
            "holt-winters.html",
            data=data_csv.to_html(classes="fixed-table", header=False, index=False),
        )
    return render_template("holt-winters.html")


@app.route("/holt-winters-define-parameter")
def holt_winters_define_parameter():
    return render_template("holt-winters-define-parameter.html")


@app.route("/svr-uni-define-parameter")
def svr_uni_define_parameter():
    return render_template("svr-uni-define-parameter.html")


@app.route("/svr-multi", methods=["GET", "POST"])
def svr_multi():
    if request.method == "POST":
        f = request.form["csv_file"]
        data = []
        with open(f) as file:
            csvfile = csv.reader(file)
            for row in csvfile:
                data.append(row)
        data = pd.DataFrame(data)

        dataset_multivariate = pd.read_csv(
            "dataset_mancanegara_kualanamu.csv", index_col="BulanTahun"
        )
        scaler = MinMaxScaler(feature_range=(0, 1))
        features_to_norm = [
            "TingkatHunianHotel(%)",
            "Events",
            "Inflasi",
            "USDToRupiah",
            "DataAktual",
        ]
        dataset_multivariate[features_to_norm] = scaler.fit_transform(
            dataset_multivariate[features_to_norm]
        )
        dataset_multivariate.columns = ["X1", "X2", "X3", "X4", "y"]

        return render_template(
            "svr-multi.html",
            data=data.to_html(classes="fixed-table", header=False, index=False),
            data_scaled=dataset_multivariate.to_html(classes="fixed-table"),
        )
    return render_template("svr-multi.html")


@app.route("/svr-multi-define-parameter")
def svr_multi_define_parameter():
    return render_template("svr-multi-define-parameter.html")


@app.route("/predict-holt-winters", methods=["GET", "POST"])
def predict_holt_winters():
    if request.method == "POST":
        req = request.form
        alpa = float(req["alpha"])
        beta = float(req["beta"])
        gamma = float(req["gamma"])

        ## Additive
        df = pd.read_csv("data.csv", sep=",")
        data_additive = df.iloc[:, 1]

        hasilPrediksiTrainingAdd = []
        smoothingAdd = []
        musimanAdd = []
        trenAdd = []
        nilaiAlpa = []
        nilaiBeta = []
        nilaiGamma = []
        nilaiMAPEAdditive = []

        nilaiAlpa.append(alpa)
        nilaiBeta.append(beta)
        nilaiGamma.append(gamma)

        nilaiSmoothingAdditive = hwf.smoothingAwal(data_additive)
        nilaiMusimanAdditive = hwf.nilaiAwalMusimanAdditive(
            data_additive, nilaiSmoothingAdditive
        )
        nilaiTrenAdditive = hwf.nilaiTrenAwal(data_additive)

        data_pakai_additive = data_additive.iloc[
            12:,
        ]
        predictAdditive = []

        for x in range(0, len(data_pakai_additive)):
            nilai_smooth_additive = hwf.smoothingKeseluruhanAdditive(
                alpa,
                data_pakai_additive,
                nilaiSmoothingAdditive,
                nilaiMusimanAdditive,
                nilaiTrenAdditive,
                x,
            )
            nilaiSmoothingAdditive.append(nilai_smooth_additive)

            nilai_tren_additive = hwf.trendSmoothingAdditive(
                beta, nilaiSmoothingAdditive, nilaiTrenAdditive, x
            )
            nilaiTrenAdditive.append(nilai_tren_additive)

            nilai_musim_additive = hwf.nilaiMusimanSmoothingAdditive(
                gamma,
                data_pakai_additive,
                nilaiSmoothingAdditive,
                nilaiMusimanAdditive,
                x,
            )
            nilaiMusimanAdditive.append(nilai_musim_additive)

            nilai_predict_additive = hwf.nilaiPredictAdditive(
                nilaiSmoothingAdditive, nilaiTrenAdditive, nilaiMusimanAdditive, x
            )
            predictAdditive.append(nilai_predict_additive)

        MAPEAdditive = 0
        errorAdditive = 0

        for i in range(0, len(data_pakai_additive)):
            totalAdditive = abs(
                (predictAdditive[i] - data_pakai_additive[i + 12]) / predictAdditive[i]
            )
            errorAdditive = errorAdditive + totalAdditive

        MAPEAdditive = errorAdditive / (len(data_pakai_additive))
        nilaiMAPEAdditive.append(MAPEAdditive)

        # MAPE Additive
        MAPEAdditive = round(MAPEAdditive * 100, 2)

        smoothingAdd.append(nilaiSmoothingAdditive)
        musimanAdd.append(nilaiMusimanAdditive)
        trenAdd.append(nilaiTrenAdditive)
        hasilPrediksiTrainingAdd.append(predictAdditive)

        smoothingadd = nilaiSmoothingAdditive[0]

        musimanadd = nilaiMusimanAdditive[0:12]
        df_musimanadd = pd.DataFrame(
            musimanadd,
            columns=["Nilai Musiman"],
            index=[
                "Januari",
                "Februari",
                "Maret",
                "April",
                "Mei",
                "Juni",
                "Juli",
                "Agustus",
                "September",
                "Oktober",
                "November",
                "Desember",
            ],
        )

        trendadd = nilaiTrenAdditive[0]

        alpaFinalAdd = nilaiAlpa[0]
        betaFinalAdd = nilaiBeta[0]
        gammaFinalAdd = nilaiGamma[0]
        smoothFinalAdd = smoothingAdd[0]
        musimanFinalAdd = musimanAdd[0]
        trenFinalAdd = trenAdd[0]

        smoothAddFinal = [smoothFinalAdd[len(data_pakai_additive)]]
        musimanAddFinal = musimanFinalAdd[(len(data_pakai_additive)) :]
        trenAddFinal = [trenFinalAdd[len(data_pakai_additive)]]
        prediksiAdd = []
        for x in range(12):
            pred = smoothAddFinal[x] + musimanAddFinal[x] + trenAddFinal[x]
            prediksiAdd.append(pred)
            smoothing = (alpaFinalAdd * (prediksiAdd[x] - musimanAddFinal[x])) + (
                (1 - alpaFinalAdd) * (smoothAddFinal[x] - trenAddFinal[x])
            )
            smoothAddFinal.append(smoothing)
            trensmoothing = (
                betaFinalAdd * ((smoothAddFinal[x + 1]) - (smoothAddFinal[x]))
            ) + ((1 - betaFinalAdd) * trenAddFinal[x])
            trenAddFinal.append(trensmoothing)
            musim = (gammaFinalAdd * (prediksiAdd[x] - smoothAddFinal[x + 1])) + (
                (1 - gammaFinalAdd) * musimanAddFinal[x]
            )
            musimanAddFinal.append(musim)

        # Prediction Additive
        data_additive = prediksiAdd
        df_prediction_add = pd.DataFrame(
            data_additive,
            columns=["Prediksi Wisatawan"],
            index=[
                "Januari 2020",
                "Februari 2020",
                "Maret 2020",
                "April 2020",
                "Mei 2020",
                "Juni 2020",
                "Juli 2020",
                "Agustus 2020",
                "September 2020",
                "Oktober 2020",
                "November 2020",
                "Desember 2020",
            ],
        )

        ## Multiplicative
        data_multiplicative = df.iloc[:, 1]

        hasilPrediksiTrainingMul = []
        smoothingMul = []
        musimanMul = []
        trenMul = []
        nilaiAlpa = []
        nilaiBeta = []
        nilaiGamma = []
        nilaiMAPEMultiplicative = []

        nilaiAlpa.append(alpa)
        nilaiBeta.append(beta)
        nilaiGamma.append(gamma)

        nilaiSmoothingMultiplicative = hwf.smoothingAwal(data_multiplicative)
        nilaiMusimanMultiplicative = hwf.nilaiAwalMusimanMultiplicative(
            data_multiplicative, nilaiSmoothingMultiplicative
        )
        nilaiTrenMultiplicative = hwf.nilaiTrenAwal(data_multiplicative)

        data_pakai_multiplicative = data_multiplicative.iloc[
            12:,
        ]
        predictMultiplicative = []

        for x in range(0, len(data_pakai_multiplicative)):
            nilai_smooth_multiplicative = hwf.smoothingKeseluruhanMultiplicative(
                alpa,
                data_pakai_multiplicative,
                nilaiSmoothingMultiplicative,
                nilaiMusimanMultiplicative,
                nilaiTrenMultiplicative,
                x,
            )
            nilaiSmoothingMultiplicative.append(nilai_smooth_multiplicative)

            nilai_tren_multiplicative = hwf.trendSmoothingMultiplicative(
                beta, nilaiSmoothingMultiplicative, nilaiTrenMultiplicative, x
            )
            nilaiTrenMultiplicative.append(nilai_tren_multiplicative)

            nilai_musim_multiplicative = hwf.nilaiMusimanSmoothingMultiplicative(
                gamma,
                data_pakai_multiplicative,
                nilaiSmoothingMultiplicative,
                nilaiMusimanMultiplicative,
                x,
            )
            nilaiMusimanMultiplicative.append(nilai_musim_multiplicative)

            nilai_predict_multiplicative = hwf.nilaiPredictMultiplicative(
                nilaiSmoothingMultiplicative,
                nilaiTrenMultiplicative,
                nilaiMusimanMultiplicative,
                x,
            )
            predictMultiplicative.append(nilai_predict_multiplicative)

        MAPEMultiplicative = 0
        errorMultiplicative = 0

        for i in range(0, len(data_pakai_multiplicative)):
            totalMultiplicative = abs(
                (predictMultiplicative[i] - data_pakai_multiplicative[i + 12])
                / predictMultiplicative[i]
            )
            errorMultiplicative = errorMultiplicative + totalMultiplicative

        MAPEMultiplicative = errorMultiplicative / (len(data_pakai_multiplicative))
        nilaiMAPEMultiplicative.append(MAPEMultiplicative)

        # MAPE Multiplicative
        MAPEMultiplicative = round(MAPEMultiplicative * 100, 2)

        smoothingMul.append(nilaiSmoothingMultiplicative)
        musimanMul.append(nilaiMusimanMultiplicative)
        trenMul.append(nilaiTrenMultiplicative)
        hasilPrediksiTrainingMul.append(predictMultiplicative)

        smoothingmul = nilaiSmoothingMultiplicative[0]

        musimanmul = nilaiMusimanMultiplicative[0:12]
        df_musimanmul = pd.DataFrame(
            musimanmul,
            columns=["Nilai Musiman"],
            index=[
                "Januari",
                "Februari",
                "Maret",
                "April",
                "Mei",
                "Juni",
                "Juli",
                "Agustus",
                "September",
                "Oktober",
                "November",
                "Desember",
            ],
        )

        trendmul = nilaiTrenMultiplicative[0]

        alpaFinalMul = nilaiAlpa[0]
        betaFinalMul = nilaiBeta[0]
        gammaFinalMul = nilaiGamma[0]
        smoothFinalMul = smoothingMul[0]
        musimanFinalMul = musimanMul[0]
        trenFinalMul = trenMul[0]

        smoothMulFinal = [smoothFinalMul[len(data_pakai_multiplicative)]]
        musimanMulFinal = musimanFinalMul[(len(data_pakai_multiplicative)) :]
        trenMulFinal = [trenFinalMul[len(data_pakai_multiplicative)]]
        prediksiMul = []
        for x in range(12):
            predik = (smoothMulFinal[x] + trenMulFinal[x]) * musimanMulFinal[x]
            prediksiMul.append(predik)
            smoothing = (alpaFinalMul * (prediksiMul[x] / musimanMulFinal[x])) + (
                (1 - alpaFinalMul) * (smoothMulFinal[x] + trenMulFinal[x])
            )
            smoothMulFinal.append(smoothing)
            trensmoothing = (
                betaFinalMul * ((smoothMulFinal[x + 1]) - (smoothMulFinal[x]))
            ) + ((1 - betaFinalMul) * trenMulFinal[x])
            trenMulFinal.append(trensmoothing)
            musim = (gammaFinalMul * (prediksiMul[x] / smoothMulFinal[x + 1])) + (
                (1 - gammaFinalMul) * musimanMulFinal[x]
            )
            musimanMulFinal.append(musim)

        data_multiplicative = prediksiMul
        df_prediction_mul = pd.DataFrame(
            data_multiplicative,
            columns=["Prediksi Wisatawan"],
            index=[
                "Januari 2020",
                "Februari 2020",
                "Maret 2020",
                "April 2020",
                "Mei 2020",
                "Juni 2020",
                "Juli 2020",
                "Agustus 2020",
                "September 2020",
                "Oktober 2020",
                "November 2020",
                "Desember 2020",
            ],
        )

        return render_template(
            "predict-holt-winters.html",
            alpha=alpa,
            beta=beta,
            gamma=gamma,
            mape_additive=MAPEAdditive,
            prediction_additive=df_prediction_add.to_html(classes="fixed-table"),
            mape_multiplicative=MAPEMultiplicative,
            prediction_multiplicative=df_prediction_mul.to_html(classes="fixed-table"),
            smoothingadd=round(smoothingadd, 2),
            trendadd=round(trendadd, 2),
            smoothingmul=round(smoothingmul, 2),
            trendmul=round(trendmul, 2),
            df_musimanadd=df_musimanadd.to_html(classes="fixed-table"),
            df_musimanmul=df_musimanmul.to_html(classes="fixed-table"),
        )

    elif request.method == "GET":
        alpa = 0.241
        beta = 0.001
        gamma = 0.724

        ## Additive
        df = pd.read_csv("data.csv", sep=",")
        data_additive = df.iloc[:, 1]

        hasilPrediksiTrainingAdd = []
        smoothingAdd = []
        musimanAdd = []
        trenAdd = []
        nilaiAlpa = []
        nilaiBeta = []
        nilaiGamma = []
        nilaiMAPEAdditive = []

        nilaiAlpa.append(alpa)
        nilaiBeta.append(beta)
        nilaiGamma.append(gamma)

        nilaiSmoothingAdditive = hwf.smoothingAwal(data_additive)
        nilaiMusimanAdditive = hwf.nilaiAwalMusimanAdditive(
            data_additive, nilaiSmoothingAdditive
        )
        nilaiTrenAdditive = hwf.nilaiTrenAwal(data_additive)

        data_pakai_additive = data_additive.iloc[
            12:,
        ]
        predictAdditive = []

        for x in range(0, len(data_pakai_additive)):
            nilai_smooth_additive = hwf.smoothingKeseluruhanAdditive(
                alpa,
                data_pakai_additive,
                nilaiSmoothingAdditive,
                nilaiMusimanAdditive,
                nilaiTrenAdditive,
                x,
            )
            nilaiSmoothingAdditive.append(nilai_smooth_additive)

            nilai_tren_additive = hwf.trendSmoothingAdditive(
                beta, nilaiSmoothingAdditive, nilaiTrenAdditive, x
            )
            nilaiTrenAdditive.append(nilai_tren_additive)

            nilai_musim_additive = hwf.nilaiMusimanSmoothingAdditive(
                gamma,
                data_pakai_additive,
                nilaiSmoothingAdditive,
                nilaiMusimanAdditive,
                x,
            )
            nilaiMusimanAdditive.append(nilai_musim_additive)

            nilai_predict_additive = hwf.nilaiPredictAdditive(
                nilaiSmoothingAdditive, nilaiTrenAdditive, nilaiMusimanAdditive, x
            )
            predictAdditive.append(nilai_predict_additive)

        MAPEAdditive = 0
        errorAdditive = 0

        for i in range(0, len(data_pakai_additive)):
            totalAdditive = abs(
                (predictAdditive[i] - data_pakai_additive[i + 12]) / predictAdditive[i]
            )
            errorAdditive = errorAdditive + totalAdditive

        MAPEAdditive = errorAdditive / (len(data_pakai_additive))
        nilaiMAPEAdditive.append(MAPEAdditive)

        # MAPE Additive
        MAPEAdditive = round(MAPEAdditive * 100, 2)

        smoothingAdd.append(nilaiSmoothingAdditive)
        musimanAdd.append(nilaiMusimanAdditive)
        trenAdd.append(nilaiTrenAdditive)
        hasilPrediksiTrainingAdd.append(predictAdditive)

        smoothingadd = nilaiSmoothingAdditive[0]

        musimanadd = nilaiMusimanAdditive[0:12]
        df_musimanadd = pd.DataFrame(
            musimanadd,
            columns=["Nilai Musiman"],
            index=[
                "Januari",
                "Februari",
                "Maret",
                "April",
                "Mei",
                "Juni",
                "Juli",
                "Agustus",
                "September",
                "Oktober",
                "November",
                "Desember",
            ],
        )

        trendadd = nilaiTrenAdditive[0]

        alpaFinalAdd = nilaiAlpa[0]
        betaFinalAdd = nilaiBeta[0]
        gammaFinalAdd = nilaiGamma[0]
        smoothFinalAdd = smoothingAdd[0]
        musimanFinalAdd = musimanAdd[0]
        trenFinalAdd = trenAdd[0]

        smoothAddFinal = [smoothFinalAdd[len(data_pakai_additive)]]
        musimanAddFinal = musimanFinalAdd[(len(data_pakai_additive)) :]
        trenAddFinal = [trenFinalAdd[len(data_pakai_additive)]]
        prediksiAdd = []
        for x in range(12):
            pred = smoothAddFinal[x] + musimanAddFinal[x] + trenAddFinal[x]
            prediksiAdd.append(pred)
            smoothing = (alpaFinalAdd * (prediksiAdd[x] - musimanAddFinal[x])) + (
                (1 - alpaFinalAdd) * (smoothAddFinal[x] - trenAddFinal[x])
            )
            smoothAddFinal.append(smoothing)
            trensmoothing = (
                betaFinalAdd * ((smoothAddFinal[x + 1]) - (smoothAddFinal[x]))
            ) + ((1 - betaFinalAdd) * trenAddFinal[x])
            trenAddFinal.append(trensmoothing)
            musim = (gammaFinalAdd * (prediksiAdd[x] - smoothAddFinal[x + 1])) + (
                (1 - gammaFinalAdd) * musimanAddFinal[x]
            )
            musimanAddFinal.append(musim)

        # Prediction Additive
        data_additive = prediksiAdd
        df_prediction_add = pd.DataFrame(
            data_additive,
            columns=["Prediksi Wisatawan"],
            index=[
                "Januari 2020",
                "Februari 2020",
                "Maret 2020",
                "April 2020",
                "Mei 2020",
                "Juni 2020",
                "Juli 2020",
                "Agustus 2020",
                "September 2020",
                "Oktober 2020",
                "November 2020",
                "Desember 2020",
            ],
        )

        ## Multiplicative
        data_multiplicative = df.iloc[:, 1]

        hasilPrediksiTrainingMul = []
        smoothingMul = []
        musimanMul = []
        trenMul = []
        nilaiAlpa = []
        nilaiBeta = []
        nilaiGamma = []
        nilaiMAPEMultiplicative = []

        nilaiAlpa.append(alpa)
        nilaiBeta.append(beta)
        nilaiGamma.append(gamma)

        nilaiSmoothingMultiplicative = hwf.smoothingAwal(data_multiplicative)
        nilaiMusimanMultiplicative = hwf.nilaiAwalMusimanMultiplicative(
            data_multiplicative, nilaiSmoothingMultiplicative
        )
        nilaiTrenMultiplicative = hwf.nilaiTrenAwal(data_multiplicative)

        data_pakai_multiplicative = data_multiplicative.iloc[
            12:,
        ]
        predictMultiplicative = []

        for x in range(0, len(data_pakai_multiplicative)):
            nilai_smooth_multiplicative = hwf.smoothingKeseluruhanMultiplicative(
                alpa,
                data_pakai_multiplicative,
                nilaiSmoothingMultiplicative,
                nilaiMusimanMultiplicative,
                nilaiTrenMultiplicative,
                x,
            )
            nilaiSmoothingMultiplicative.append(nilai_smooth_multiplicative)

            nilai_tren_multiplicative = hwf.trendSmoothingMultiplicative(
                beta, nilaiSmoothingMultiplicative, nilaiTrenMultiplicative, x
            )
            nilaiTrenMultiplicative.append(nilai_tren_multiplicative)

            nilai_musim_multiplicative = hwf.nilaiMusimanSmoothingMultiplicative(
                gamma,
                data_pakai_multiplicative,
                nilaiSmoothingMultiplicative,
                nilaiMusimanMultiplicative,
                x,
            )
            nilaiMusimanMultiplicative.append(nilai_musim_multiplicative)

            nilai_predict_multiplicative = hwf.nilaiPredictMultiplicative(
                nilaiSmoothingMultiplicative,
                nilaiTrenMultiplicative,
                nilaiMusimanMultiplicative,
                x,
            )
            predictMultiplicative.append(nilai_predict_multiplicative)

        MAPEMultiplicative = 0
        errorMultiplicative = 0

        for i in range(0, len(data_pakai_multiplicative)):
            totalMultiplicative = abs(
                (predictMultiplicative[i] - data_pakai_multiplicative[i + 12])
                / predictMultiplicative[i]
            )
            errorMultiplicative = errorMultiplicative + totalMultiplicative

        MAPEMultiplicative = errorMultiplicative / (len(data_pakai_multiplicative))
        nilaiMAPEMultiplicative.append(MAPEMultiplicative)

        # MAPE Multiplicative
        MAPEMultiplicative = round(MAPEMultiplicative * 100, 2)

        smoothingMul.append(nilaiSmoothingMultiplicative)
        musimanMul.append(nilaiMusimanMultiplicative)
        trenMul.append(nilaiTrenMultiplicative)
        hasilPrediksiTrainingMul.append(predictMultiplicative)

        smoothingmul = nilaiSmoothingMultiplicative[0]

        musimanmul = nilaiMusimanMultiplicative[0:12]
        df_musimanmul = pd.DataFrame(
            musimanmul,
            columns=["Nilai Musiman"],
            index=[
                "Januari",
                "Februari",
                "Maret",
                "April",
                "Mei",
                "Juni",
                "Juli",
                "Agustus",
                "September",
                "Oktober",
                "November",
                "Desember",
            ],
        )

        trendmul = nilaiTrenMultiplicative[0]

        alpaFinalMul = nilaiAlpa[0]
        betaFinalMul = nilaiBeta[0]
        gammaFinalMul = nilaiGamma[0]
        smoothFinalMul = smoothingMul[0]
        musimanFinalMul = musimanMul[0]
        trenFinalMul = trenMul[0]

        smoothMulFinal = [smoothFinalMul[len(data_pakai_multiplicative)]]
        musimanMulFinal = musimanFinalMul[(len(data_pakai_multiplicative)) :]
        trenMulFinal = [trenFinalMul[len(data_pakai_multiplicative)]]
        prediksiMul = []
        for x in range(12):
            predik = (smoothMulFinal[x] + trenMulFinal[x]) * musimanMulFinal[x]
            prediksiMul.append(predik)
            smoothing = (alpaFinalMul * (prediksiMul[x] / musimanMulFinal[x])) + (
                (1 - alpaFinalMul) * (smoothMulFinal[x] + trenMulFinal[x])
            )
            smoothMulFinal.append(smoothing)
            trensmoothing = (
                betaFinalMul * ((smoothMulFinal[x + 1]) - (smoothMulFinal[x]))
            ) + ((1 - betaFinalMul) * trenMulFinal[x])
            trenMulFinal.append(trensmoothing)
            musim = (gammaFinalMul * (prediksiMul[x] / smoothMulFinal[x + 1])) + (
                (1 - gammaFinalMul) * musimanMulFinal[x]
            )
            musimanMulFinal.append(musim)

        data_multiplicative = prediksiMul
        df_prediction_mul = pd.DataFrame(
            data_multiplicative,
            columns=["Prediksi Wisatawan"],
            index=[
                "Januari 2020",
                "Februari 2020",
                "Maret 2020",
                "April 2020",
                "Mei 2020",
                "Juni 2020",
                "Juli 2020",
                "Agustus 2020",
                "September 2020",
                "Oktober 2020",
                "November 2020",
                "Desember 2020",
            ],
        )

        return render_template(
            "predict-holt-winters.html",
            alpha=alpa,
            beta=beta,
            gamma=gamma,
            mape_additive=MAPEAdditive,
            prediction_additive=df_prediction_add.to_html(classes="fixed-table"),
            mape_multiplicative=MAPEMultiplicative,
            prediction_multiplicative=df_prediction_mul.to_html(classes="fixed-table"),
            smoothingadd=round(smoothingadd, 2),
            trendadd=round(trendadd, 2),
            smoothingmul=round(smoothingmul, 2),
            trendmul=round(trendmul, 2),
            df_musimanadd=df_musimanadd.to_html(classes="fixed-table"),
            df_musimanmul=df_musimanmul.to_html(classes="fixed-table"),
        )
    return render_template("predict-holt-winters.html")


@app.route("/svr-uni", methods=["GET", "POST"])
def svr_uni():
    if request.method == "POST":
        f = request.form["csv_file"]
        data = []
        with open(f) as file:
            csvfile = csv.reader(file)
            for row in csvfile:
                data.append(row)
        data = pd.DataFrame(data)

        dataset = pd.read_csv("dataset_univariate.csv", index_col="BulanTahun")
        dataset = dataset[["DataAktual"]]
        scaler = MinMaxScaler(feature_range=(0, 1))
        col_to_norm = ["DataAktual"]
        dataset[col_to_norm] = scaler.fit_transform(dataset[col_to_norm])
        reorder_cols = ["y_4", "y_3", "y_2", "y_1", "DataAktual"]
        df_reframe = svrf.reframe_to_supervised(dataset)
        df_reframe = df_reframe.reindex(columns=reorder_cols)
        dataset_univariate = df_reframe.dropna()

        return render_template(
            "svr-uni.html",
            data=data.to_html(classes="fixed-table", header=False, index=False),
            data_scaled=dataset[col_to_norm].to_html(classes="fixed-table"),
            data_reframed=dataset_univariate.to_html(classes="fixed-table"),
        )
    return render_template("svr-uni.html")


@app.route("/predict-svr-uni", methods=["GET", "POST"])
def predict_svr_uni():
    if request.method == "POST":
        dataset = pd.read_csv("dataset_univariate.csv", index_col="BulanTahun")
        dataset = dataset[["DataAktual"]]
        scaler = MinMaxScaler(feature_range=(0, 1))
        col_to_norm = ["DataAktual"]
        dataset[col_to_norm] = scaler.fit_transform(dataset[col_to_norm])
        reorder_cols = ["y_4", "y_3", "y_2", "y_1", "DataAktual"]
        df_reframe = svrf.reframe_to_supervised(dataset)
        df_reframe = df_reframe.reindex(columns=reorder_cols)
        dataset_univariate = df_reframe.dropna()
        dataset_univariate.columns = ["y_4", "y_3", "y_2", "y_1", "y"]
        features = ["y_4", "y_3", "y_2", "y_1"]
        target = ["y"]
        X = dataset_univariate[features]
        y = dataset_univariate[target]

        # Splitting Data
        train_uni = int(request.form["train-uni"])
        test_uni = int(request.form["test-uni"])
        split_dataset = int((train_uni / 100) * len(X))
        X_train, X_test = X[:split_dataset], X[split_dataset:]
        y_train, y_test = y[:split_dataset], y[split_dataset:]

        # Best Parameter Values based on Researchers (TA1920-02/us)
        # C=0.5 s/d 50, cLR=0.01, epsilon=0.001, lambda=0.00001 s/d 0.01, sigma=0.5, iterasi=50
        C = 10
        cLR = 0.01
        epsilon = 0.001
        _lambda = 0.01
        sigma = 0.5
        iteration = int(request.form["iteration-uni"])

        df_distance_train = svrf.calculate_distance_train(X_train)

        # Formula: exp(-(perhitungan jarak)/(2*(sigma^2))
        def calculate_kernel(data):
            # i, j index for data
            df_kernel = [[] for i in range(len(data.index))]
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    kernel = math.exp(-(data.values[i, j]) / (2 * pow(sigma, 2)))
                    df_kernel[i].append(kernel)
            df_kernel = pd.DataFrame(df_kernel)
            return df_kernel

        df_kernel_train = calculate_kernel(df_distance_train)

        # Formula: K(xi,xj) + 𝝺^2
        def calculate_hessian(data):
            # i, j index for data
            df_hessian = [[] for i in range(len(data.index))]
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    hessian = data.values[i, j] + pow(_lambda, 2)
                    df_hessian[i].append(hessian)
            df_hessian = pd.DataFrame(df_hessian)
            return df_hessian

        df_hessian_train = calculate_hessian(df_kernel_train)

        gamma = round(cLR / max(df_hessian_train.max()), 3)

        for i in range(iteration):
            data_train = y_train
            data_hessian = df_hessian_train

            if i == 0:
                df_alpha = svrf.init_alpha(data_train)
                df_alpha_star = svrf.init_alpha_star(data_train)
            elif i > 0:
                df_alpha = df_update_alpha
                df_alpha_star = df_update_alpha_star

            df_multipliers = svrf.alpha_star_min_alpha(df_alpha_star, df_alpha)
            df_multipliers_cross_hessian = svrf.multipliers_cross_hessian(
                df_multipliers, data_hessian
            )
            df_error = svrf.y_min_multipliers_cross_hessian(
                data_train, df_multipliers_cross_hessian
            )
            df_epsilon = svrf.epsilon_to_df(data_train, epsilon)
            df_gamma = svrf.gamma_to_df(data_train, gamma)
            df_c = svrf.c_to_df(data_train, C)

            df_error_min_epsilon = svrf.error_min_epsilon(df_error, df_epsilon)
            df_gamma_cross_error_min_epsilon = svrf.gamma_cross_error_min_epsilon(
                df_gamma, df_error_min_epsilon
            )
            df_c_min_alpha_star = svrf.c_min_alpha_star(df_c, df_alpha_star)
            df_minus_alpha_star = svrf.convert_to_minus(df_alpha_star)
            df_max_alpha_star = svrf.data_maximum_alpha_star(
                df_gamma_cross_error_min_epsilon, df_minus_alpha_star
            )
            df_delta_alpha_star = svrf.data_minimum_alpha_star(
                df_max_alpha_star, df_c_min_alpha_star
            )

            df_minus_error = svrf.convert_to_minus(df_error)
            df_minus_alpha = svrf.convert_to_minus(df_alpha)
            df_c_min_alpha = svrf.c_min_alpha(df_c, df_alpha)
            df_min_error_min_epsilon = svrf.min_error_min_epsilon(
                df_minus_error, df_epsilon
            )
            df_gamma_cross_min_error_min_epsilon = svrf.gamma_cross_min_error_min_epsilon(
                df_gamma, df_min_error_min_epsilon
            )
            df_max_alpha = svrf.data_maximum_alpha(
                df_gamma_cross_min_error_min_epsilon, df_minus_alpha
            )
            df_delta_alpha = svrf.data_minimum_alpha(df_max_alpha, df_c_min_alpha)

            df_update_alpha = svrf.update_alpha(df_delta_alpha, df_alpha)
            df_update_alpha_star = svrf.update_alpha_star(
                df_delta_alpha_star, df_alpha_star
            )

            concat_df_train_iteration = [
                df_error,
                df_delta_alpha_star,
                df_delta_alpha,
                df_update_alpha_star,
                df_update_alpha,
            ]
            df_train_iteration = pd.concat(concat_df_train_iteration, axis=1)

            # Check condition
            ## max(abs(delta_alpha_star) < epsilon and max(abs(delta_alpha) < epsilon ==> Stop Iteration
            abs_delta_alpha_star = abs(df_delta_alpha_star)
            abs_delta_alpha = abs(df_delta_alpha)
            maximum_delta_alpha_star = (abs_delta_alpha_star.max()).max()
            maximum_delta_alpha = (abs_delta_alpha.max()).max()
            if (
                maximum_delta_alpha_star < epsilon and maximum_delta_alpha < epsilon
            ) and i < iteration:
                break

        df_updated_alpha_star = df_train_iteration[["update_alpha_star"]]
        df_updated_alpha = df_train_iteration[["update_alpha"]]
        df_updated_multipliers = svrf.alpha_star_min_alpha(
            df_updated_alpha_star, df_updated_alpha
        )

        # Denormalized y and f(X)
        df_univarite = pd.read_csv("dataset_univariate.csv", index_col="BulanTahun")
        y_actual = df_univarite[["DataAktual"]]
        df_distance_test = svrf.calcute_distance_test(X_test, X_train)
        df_kernel_test = calculate_kernel(df_distance_test)
        df_hessian_test = calculate_hessian(df_kernel_test)

        df_regression_function_test = svrf.regression_function(
            df_updated_multipliers, df_hessian_test
        )
        df_denormalized_y_test = svrf.denormalized_y_actual(y_test, y_actual)
        df_denormalized_y_pred_test = svrf.denormalized_y_pred(
            df_regression_function_test, y_actual
        )

        # df_prediction = pd.DataFrame(df_denormalized_y_pred_test.values, index=[y_test.index], columns=['Prediction'])
        df_predict_y_test_and_y_actual = pd.concat(
            [df_denormalized_y_test, df_denormalized_y_pred_test], axis=1
        )
        df_predict_y_test_and_y_actual = pd.DataFrame(
            df_predict_y_test_and_y_actual.values,
            index=[y_test.index],
            columns=["Data Aktual", "Prediksi"],
        )

        mape_test = svrf.calculate_mape(
            df_denormalized_y_test, df_denormalized_y_pred_test
        )

        # Feature 2020
        df_feature_2020 = pd.read_csv("feature_univariate.csv", index_col="BulanTahun")
        y_actual_feature_2020 = pd.read_csv(
            "feature_univariate.csv", index_col="BulanTahun"
        )
        column_to_norm = ["DataAktual"]
        df_feature_2020[column_to_norm] = scaler.fit_transform(
            df_feature_2020[column_to_norm]
        )
        # Reframe Feature 2020 to supervised forms
        df_reframe_feature2020 = svrf.reframe_to_supervised(df_feature_2020)
        df_reframe_feature2020 = df_reframe_feature2020.reindex(columns=reorder_cols)
        df_feature_2020 = df_reframe_feature2020.dropna()
        df_feature_2020.columns = ["y_4", "y_3", "y_2", "y_1", "y"]
        X_feature_2020 = df_feature_2020[features]
        y_feature_2020 = df_feature_2020[target]
        # Calculation of Distance between Data Train and Feature 2020
        df_distance_feature_2020 = svrf.calcute_distance_test(X_feature_2020, X_train)
        # Calculation of Kernel
        df_kernel_feature_2020 = calculate_kernel(df_distance_feature_2020)
        # Calculation of Matrix Hessian
        df_hessian_feature_2020 = calculate_hessian(df_kernel_feature_2020)
        # y_feature_2020
        df_regression_function_feature_2020 = svrf.regression_function(
            df_updated_multipliers, df_hessian_feature_2020
        )
        # Denormalized y_feature_2020
        df_denormalized_y_feature_2020 = svrf.denormalized_y_pred(
            df_regression_function_feature_2020, y_actual_feature_2020
        )
        # Denormalized y_actual_2020
        df_denormalized_y_actual_2020 = svrf.denormalized_y_actual(
            y_feature_2020, y_actual_feature_2020
        )

        index_feature_2020 = [
            "June 2019",
            "July 2019",
            "August 2019",
            "September 2019",
            "October 2019",
            "November 2019",
            "December 2019",
            "January 2020",
            "February 2020",
        ]
        df_prediction_2020 = pd.concat(
            [df_denormalized_y_actual_2020, df_denormalized_y_feature_2020], axis=1
        )
        df_predict_feature_2020 = pd.DataFrame(
            df_prediction_2020.values,
            index=[index_feature_2020],
            columns=["Jumlah Wisatawan", "Prediksi 2020"],
        )

        return render_template(
            "predict-svr-uni.html",
            predictionUni=df_predict_y_test_and_y_actual.to_html(classes="fixed-table"),
            trainUni=train_uni,
            testUni=test_uni,
            clrUni=cLR,
            cUni=C,
            epsilonUni=epsilon,
            lambdaUni=_lambda,
            sigmaUni=sigma,
            iterationUni=iteration,
            mape_test=mape_test,
            prediction_2020=df_predict_feature_2020.to_html(classes="fixed-table"),
        )
    return render_template("predict-svr-uni.html")


@app.route("/predict-svr-uni-define-parameter", methods=["GET", "POST"])
def predict_svr_uni_define():
    if request.method == "POST":
        dataset = pd.read_csv("dataset_univariate.csv", index_col="BulanTahun")
        dataset = dataset[["DataAktual"]]
        scaler = MinMaxScaler(feature_range=(0, 1))
        col_to_norm = ["DataAktual"]
        dataset[col_to_norm] = scaler.fit_transform(dataset[col_to_norm])
        reorder_cols = ["y_4", "y_3", "y_2", "y_1", "DataAktual"]
        df_reframe = svrf.reframe_to_supervised(dataset)
        df_reframe = df_reframe.reindex(columns=reorder_cols)
        dataset_univariate = df_reframe.dropna()
        dataset_univariate.columns = ["y_4", "y_3", "y_2", "y_1", "y"]
        features = ["y_4", "y_3", "y_2", "y_1"]
        target = ["y"]
        X = dataset_univariate[features]
        y = dataset_univariate[target]

        # Splitting Data
        train_uni = int(request.form["train-uni"])
        test_uni = int(request.form["test-uni"])

        split_dataset = int((train_uni / 100) * len(X))
        X_train, X_test = X[:split_dataset], X[split_dataset:]
        y_train, y_test = y[:split_dataset], y[split_dataset:]

        cLR = float(request.form["clr-uni"])
        C = float(request.form["c-uni"])
        epsilon = float(request.form["epsilon-uni"])
        _lambda = float(request.form["lambda-uni"])
        sigma = float(request.form["sigma-uni"])
        iteration = int(request.form["iteration-uni"])

        df_distance_train = svrf.calculate_distance_train(X_train)

        def calculate_kernel(data):
            # i, j index for data
            df_kernel = [[] for i in range(len(data.index))]
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    kernel = math.exp(-(data.values[i, j]) / (2 * pow(sigma, 2)))
                    df_kernel[i].append(kernel)
            df_kernel = pd.DataFrame(df_kernel)
            return df_kernel

        df_kernel_train = calculate_kernel(df_distance_train)

        def calculate_hessian(data):
            # i, j index for data
            df_hessian = [[] for i in range(len(data.index))]
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    hessian = data.values[i, j] + pow(_lambda, 2)
                    df_hessian[i].append(hessian)
            df_hessian = pd.DataFrame(df_hessian)
            return df_hessian

        df_hessian_train = calculate_hessian(df_kernel_train)

        gamma = round(cLR / max(df_hessian_train.max()), 3)

        for i in range(iteration):
            data_train = y_train
            data_hessian = df_hessian_train

            if i == 0:
                df_alpha = svrf.init_alpha(data_train)
                df_alpha_star = svrf.init_alpha_star(data_train)
            elif i > 0:
                df_alpha = df_update_alpha
                df_alpha_star = df_update_alpha_star

            df_multipliers = svrf.alpha_star_min_alpha(df_alpha_star, df_alpha)
            df_multipliers_cross_hessian = svrf.multipliers_cross_hessian(
                df_multipliers, data_hessian
            )
            df_error = svrf.y_min_multipliers_cross_hessian(
                data_train, df_multipliers_cross_hessian
            )
            df_epsilon = svrf.epsilon_to_df(data_train, epsilon)
            df_gamma = svrf.gamma_to_df(data_train, gamma)
            df_c = svrf.c_to_df(data_train, C)

            df_error_min_epsilon = svrf.error_min_epsilon(df_error, df_epsilon)
            df_gamma_cross_error_min_epsilon = svrf.gamma_cross_error_min_epsilon(
                df_gamma, df_error_min_epsilon
            )
            df_c_min_alpha_star = svrf.c_min_alpha_star(df_c, df_alpha_star)
            df_minus_alpha_star = svrf.convert_to_minus(df_alpha_star)
            df_max_alpha_star = svrf.data_maximum_alpha_star(
                df_gamma_cross_error_min_epsilon, df_minus_alpha_star
            )
            df_delta_alpha_star = svrf.data_minimum_alpha_star(
                df_max_alpha_star, df_c_min_alpha_star
            )

            df_minus_error = svrf.convert_to_minus(df_error)
            df_minus_alpha = svrf.convert_to_minus(df_alpha)
            df_c_min_alpha = svrf.c_min_alpha(df_c, df_alpha)
            df_min_error_min_epsilon = svrf.min_error_min_epsilon(
                df_minus_error, df_epsilon
            )
            df_gamma_cross_min_error_min_epsilon = svrf.gamma_cross_min_error_min_epsilon(
                df_gamma, df_min_error_min_epsilon
            )
            df_max_alpha = svrf.data_maximum_alpha(
                df_gamma_cross_min_error_min_epsilon, df_minus_alpha
            )
            df_delta_alpha = svrf.data_minimum_alpha(df_max_alpha, df_c_min_alpha)

            df_update_alpha = svrf.update_alpha(df_delta_alpha, df_alpha)
            df_update_alpha_star = svrf.update_alpha_star(
                df_delta_alpha_star, df_alpha_star
            )

            concat_df_train_iteration = [
                df_error,
                df_delta_alpha_star,
                df_delta_alpha,
                df_update_alpha_star,
                df_update_alpha,
            ]
            df_train_iteration = pd.concat(concat_df_train_iteration, axis=1)

            abs_delta_alpha_star = abs(df_delta_alpha_star)
            abs_delta_alpha = abs(df_delta_alpha)
            maximum_delta_alpha_star = (abs_delta_alpha_star.max()).max()
            maximum_delta_alpha = (abs_delta_alpha.max()).max()
            if (
                maximum_delta_alpha_star < epsilon and maximum_delta_alpha < epsilon
            ) and i < iteration:
                break

        df_updated_alpha_star = df_train_iteration[["update_alpha_star"]]
        df_updated_alpha = df_train_iteration[["update_alpha"]]
        df_updated_multipliers = svrf.alpha_star_min_alpha(
            df_updated_alpha_star, df_updated_alpha
        )

        # Denormalized y and f(X)
        df_univarite = pd.read_csv("dataset_univariate.csv", index_col="BulanTahun")
        y_actual = df_univarite[["DataAktual"]]

        df_distance_test = svrf.calcute_distance_test(X_test, X_train)
        df_kernel_test = calculate_kernel(df_distance_test)
        df_hessian_test = calculate_hessian(df_kernel_test)

        df_regression_function_test = svrf.regression_function(
            df_updated_multipliers, df_hessian_test
        )
        df_denormalized_y_test = svrf.denormalized_y_actual(y_test, y_actual)
        df_denormalized_y_pred_test = svrf.denormalized_y_pred(
            df_regression_function_test, y_actual
        )

        # df_prediction = pd.DataFrame(df_denormalized_y_pred_test.values, index=[y_test.index], columns=['Prediction'])
        df_predict_y_test_and_y_actual = pd.concat(
            [df_denormalized_y_test, df_denormalized_y_pred_test], axis=1
        )
        df_predict_y_test_and_y_actual = pd.DataFrame(
            df_predict_y_test_and_y_actual.values,
            index=[y_test.index],
            columns=["Data Aktual", "Prediksi"],
        )

        mape_test = svrf.calculate_mape(
            df_denormalized_y_test, df_denormalized_y_pred_test
        )

        # Feature 2020
        df_feature_2020 = pd.read_csv("feature_univariate.csv", index_col="BulanTahun")
        y_actual_feature_2020 = pd.read_csv(
            "feature_univariate.csv", index_col="BulanTahun"
        )
        column_to_norm = ["DataAktual"]
        df_feature_2020[column_to_norm] = scaler.fit_transform(
            df_feature_2020[column_to_norm]
        )
        # Reframe Feature 2020 to supervised forms
        df_reframe_feature2020 = svrf.reframe_to_supervised(df_feature_2020)
        df_reframe_feature2020 = df_reframe_feature2020.reindex(columns=reorder_cols)
        df_feature_2020 = df_reframe_feature2020.dropna()
        df_feature_2020.columns = ["y_4", "y_3", "y_2", "y_1", "y"]
        X_feature_2020 = df_feature_2020[features]
        y_feature_2020 = df_feature_2020[target]
        # Calculation of Distance between Data Train and Feature 2020
        df_distance_feature_2020 = svrf.calcute_distance_test(X_feature_2020, X_train)
        # Calculation of Kernel
        df_kernel_feature_2020 = calculate_kernel(df_distance_feature_2020)
        # Calculation of Matrix Hessian
        df_hessian_feature_2020 = calculate_hessian(df_kernel_feature_2020)
        # y_feature_2020
        df_regression_function_feature_2020 = svrf.regression_function(
            df_updated_multipliers, df_hessian_feature_2020
        )
        # Denormalized y_feature_2020
        df_denormalized_y_feature_2020 = svrf.denormalized_y_pred(
            df_regression_function_feature_2020, y_actual_feature_2020
        )
        # Denormalized y_actual_2020
        df_denormalized_y_actual_2020 = svrf.denormalized_y_actual(
            y_feature_2020, y_actual_feature_2020
        )

        index_feature_2020 = [
            "June 2019",
            "July 2019",
            "August 2019",
            "September 2019",
            "October 2019",
            "November 2019",
            "December 2019",
            "January 2020",
            "February 2020",
        ]
        df_prediction_2020 = pd.concat(
            [df_denormalized_y_actual_2020, df_denormalized_y_feature_2020], axis=1
        )
        df_predict_feature_2020 = pd.DataFrame(
            df_prediction_2020.values,
            index=[index_feature_2020],
            columns=["Jumlah Wisatawan", "Prediksi 2020"],
        )

        return render_template(
            "predict-svr-uni.html",
            predictionUni=df_predict_y_test_and_y_actual.to_html(classes="fixed-table"),
            trainUni=train_uni,
            testUni=test_uni,
            clrUni=cLR,
            cUni=C,
            epsilonUni=epsilon,
            lambdaUni=_lambda,
            sigmaUni=sigma,
            iterationUni=iteration,
            mape_test=mape_test,
            prediction_2020=df_predict_feature_2020.to_html(classes="fixed-table"),
        )
    return render_template("predict-svr-uni.html")


@app.route("/predict-svr-multi", methods=["GET", "POST"])
def predict_svr_multi():
    if request.method == "POST":
        dataset_multivariate = pd.read_csv(
            "dataset_mancanegara_kualanamu.csv", index_col="BulanTahun"
        )
        scaler = MinMaxScaler(feature_range=(0, 1))
        features_to_norm = [
            "TingkatHunianHotel(%)",
            "Events",
            "Inflasi",
            "USDToRupiah",
            "DataAktual",
        ]
        dataset_multivariate[features_to_norm] = scaler.fit_transform(
            dataset_multivariate[features_to_norm]
        )
        dataset_multivariate.columns = ["X1", "X2", "X3", "X4", "y"]
        features = ["X1", "X2", "X3", "X4"]
        target = ["y"]
        X = dataset_multivariate[features]
        y = dataset_multivariate[target]

        # Splitting Data
        train_multi = int(request.form["train-multi"])
        test_multi = int(request.form["test-multi"])
        split_dataset = int((train_multi / 100) * len(X))
        X_train, X_test = X[:split_dataset], X[split_dataset:]
        y_train, y_test = y[:split_dataset], y[split_dataset:]

        # Best Parameter Values based on Researchers (TA1920-02/us)
        # C=0.5 s/d 50, cLR=0.01, epsilon=0.001, lambda=0.1, sigma=1, iterasi=50
        cLR = 0.01
        C = 50
        epsilon = 0.001
        _lambda = 0.1
        sigma = 1
        iteration = int(request.form["iteration-multi"])

        df_distance_train = svrf.calculate_distance_train(X_train)

        def calculate_kernel(data):
            # i, j index for data
            df_kernel = [[] for i in range(len(data.index))]
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    kernel = math.exp(-(data.values[i, j]) / (2 * pow(sigma, 2)))
                    df_kernel[i].append(kernel)
            df_kernel = pd.DataFrame(df_kernel)
            return df_kernel

        df_kernel_train = calculate_kernel(df_distance_train)

        def calculate_hessian(data):
            # i, j index for data
            df_hessian = [[] for i in range(len(data.index))]
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    hessian = data.values[i, j] + pow(_lambda, 2)
                    df_hessian[i].append(hessian)
            df_hessian = pd.DataFrame(df_hessian)
            return df_hessian

        df_hessian_train = calculate_hessian(df_kernel_train)

        gamma = round(cLR / max(df_hessian_train.max()), 3)

        for i in range(iteration):
            data_train = y_train
            data_hessian = df_hessian_train

            if i == 0:
                df_alpha = svrf.init_alpha(data_train)
                df_alpha_star = svrf.init_alpha_star(data_train)
            elif i > 0:
                df_alpha = df_update_alpha
                df_alpha_star = df_update_alpha_star

            df_multipliers = svrf.alpha_star_min_alpha(df_alpha_star, df_alpha)
            df_multipliers_cross_hessian = svrf.multipliers_cross_hessian(
                df_multipliers, data_hessian
            )
            df_error = svrf.y_min_multipliers_cross_hessian(
                data_train, df_multipliers_cross_hessian
            )
            df_epsilon = svrf.epsilon_to_df(data_train, epsilon)
            df_gamma = svrf.gamma_to_df(data_train, gamma)
            df_c = svrf.c_to_df(data_train, C)

            df_error_min_epsilon = svrf.error_min_epsilon(df_error, df_epsilon)
            df_gamma_cross_error_min_epsilon = svrf.gamma_cross_error_min_epsilon(
                df_gamma, df_error_min_epsilon
            )
            df_c_min_alpha_star = svrf.c_min_alpha_star(df_c, df_alpha_star)
            df_minus_alpha_star = svrf.convert_to_minus(df_alpha_star)
            df_max_alpha_star = svrf.data_maximum_alpha_star(
                df_gamma_cross_error_min_epsilon, df_minus_alpha_star
            )
            df_delta_alpha_star = svrf.data_minimum_alpha_star(
                df_max_alpha_star, df_c_min_alpha_star
            )

            df_minus_error = svrf.convert_to_minus(df_error)
            df_minus_alpha = svrf.convert_to_minus(df_alpha)
            df_c_min_alpha = svrf.c_min_alpha(df_c, df_alpha)
            df_min_error_min_epsilon = svrf.min_error_min_epsilon(
                df_minus_error, df_epsilon
            )
            df_gamma_cross_min_error_min_epsilon = svrf.gamma_cross_min_error_min_epsilon(
                df_gamma, df_min_error_min_epsilon
            )
            df_max_alpha = svrf.data_maximum_alpha(
                df_gamma_cross_min_error_min_epsilon, df_minus_alpha
            )
            df_delta_alpha = svrf.data_minimum_alpha(df_max_alpha, df_c_min_alpha)

            df_update_alpha = svrf.update_alpha(df_delta_alpha, df_alpha)
            df_update_alpha_star = svrf.update_alpha_star(
                df_delta_alpha_star, df_alpha_star
            )

            concat_df_train_iteration = [
                df_error,
                df_delta_alpha_star,
                df_delta_alpha,
                df_update_alpha_star,
                df_update_alpha,
            ]
            df_train_iteration = pd.concat(concat_df_train_iteration, axis=1)

            # Check condition
            ## max(abs(delta_alpha_star) < epsilon and max(abs(delta_alpha) < epsilon ==> Stop Iteration
            abs_delta_alpha_star = abs(df_delta_alpha_star)
            abs_delta_alpha = abs(df_delta_alpha)
            maximum_delta_alpha_star = (abs_delta_alpha_star.max()).max()
            maximum_delta_alpha = (abs_delta_alpha.max()).max()
            if (
                maximum_delta_alpha_star < epsilon and maximum_delta_alpha < epsilon
            ) and i < iteration:
                break

        df_updated_alpha_star = df_train_iteration[["update_alpha_star"]]
        df_updated_alpha = df_train_iteration[["update_alpha"]]
        df_updated_multipliers = svrf.alpha_star_min_alpha(
            df_updated_alpha_star, df_updated_alpha
        )

        df_multivarite = pd.read_csv(
            "dataset_mancanegara_kualanamu.csv", index_col="BulanTahun"
        )
        y_actual = df_multivarite[["DataAktual"]]

        df_distance_test = svrf.calcute_distance_test(X_test, X_train)
        df_kernel_test = calculate_kernel(df_distance_test)
        df_hessian_test = calculate_hessian(df_kernel_test)

        df_regression_function_test = svrf.regression_function(
            df_updated_multipliers, df_hessian_test
        )
        df_denormalized_y_test = svrf.denormalized_y_actual(y_test, y_actual)
        df_denormalized_y_pred_test = svrf.denormalized_y_pred(
            df_regression_function_test, y_actual
        )

        # df_prediction = pd.DataFrame(df_denormalized_y_pred_test.values, index=[y_test.index], columns=['Prediction'])
        df_predict_y_test_and_y_actual = pd.concat(
            [df_denormalized_y_test, df_denormalized_y_pred_test], axis=1
        )
        df_predict_y_test_and_y_actual = pd.DataFrame(
            df_predict_y_test_and_y_actual.values,
            index=[y_test.index],
            columns=["Data Aktual", "Prediksi"],
        )

        mape_test = svrf.calculate_mape(
            df_denormalized_y_test, df_denormalized_y_pred_test
        )

        return render_template(
            "predict-svr-multi.html",
            predictionMulti=df_predict_y_test_and_y_actual.to_html(
                classes="fixed-table"
            ),
            trainMulti=train_multi,
            testMulti=test_multi,
            clrMulti=cLR,
            cMulti=C,
            epsilonMulti=epsilon,
            lambdaMulti=_lambda,
            sigmaMulti=sigma,
            iterationMulti=iteration,
            mape_test=mape_test,
        )
    return render_template("predict-svr-multi.html")


@app.route("/predict-svr-multi-define-parameter", methods=["GET", "POST"])
def predict_svr_multi_define():
    if request.method == "POST":
        dataset_multivariate = pd.read_csv(
            "dataset_mancanegara_kualanamu.csv", index_col="BulanTahun"
        )
        scaler = MinMaxScaler(feature_range=(0, 1))
        features_to_norm = [
            "TingkatHunianHotel(%)",
            "Events",
            "Inflasi",
            "USDToRupiah",
            "DataAktual",
        ]
        dataset_multivariate[features_to_norm] = scaler.fit_transform(
            dataset_multivariate[features_to_norm]
        )
        dataset_multivariate.columns = ["X1", "X2", "X3", "X4", "y"]
        features = ["X1", "X2", "X3", "X4"]
        target = ["y"]
        X = dataset_multivariate[features]
        y = dataset_multivariate[target]

        # Splitting Data
        train_multi = int(request.form["train-multi"])
        test_multi = int(request.form["test-multi"])
        split_dataset = int((train_multi / 100) * len(X))
        X_train, X_test = X[:split_dataset], X[split_dataset:]
        y_train, y_test = y[:split_dataset], y[split_dataset:]

        cLR = float(request.form["clr-multi"])
        C = float(request.form["c-multi"])
        epsilon = float(request.form["epsilon-multi"])
        _lambda = float(request.form["lambda-multi"])
        sigma = float(request.form["sigma-multi"])
        iteration = int(request.form["iteration-multi"])

        df_distance_train = svrf.calculate_distance_train(X_train)

        def calculate_kernel(data):
            # i, j index for data
            df_kernel = [[] for i in range(len(data.index))]
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    kernel = math.exp(-(data.values[i, j]) / (2 * pow(sigma, 2)))
                    df_kernel[i].append(kernel)
            df_kernel = pd.DataFrame(df_kernel)
            return df_kernel

        df_kernel_train = calculate_kernel(df_distance_train)

        def calculate_hessian(data):
            # i, j index for data
            df_hessian = [[] for i in range(len(data.index))]
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    hessian = data.values[i, j] + pow(_lambda, 2)
                    df_hessian[i].append(hessian)
            df_hessian = pd.DataFrame(df_hessian)
            return df_hessian

        df_hessian_train = calculate_hessian(df_kernel_train)

        gamma = round(cLR / max(df_hessian_train.max()), 3)

        for i in range(iteration):
            data_train = y_train
            data_hessian = df_hessian_train

            if i == 0:
                df_alpha = svrf.init_alpha(data_train)
                df_alpha_star = svrf.init_alpha_star(data_train)
            elif i > 0:
                df_alpha = df_update_alpha
                df_alpha_star = df_update_alpha_star

            df_multipliers = svrf.alpha_star_min_alpha(df_alpha_star, df_alpha)
            df_multipliers_cross_hessian = svrf.multipliers_cross_hessian(
                df_multipliers, data_hessian
            )
            df_error = svrf.y_min_multipliers_cross_hessian(
                data_train, df_multipliers_cross_hessian
            )

            df_epsilon = svrf.epsilon_to_df(data_train, epsilon)
            df_gamma = svrf.gamma_to_df(data_train, gamma)
            df_c = svrf.c_to_df(data_train, C)

            df_error_min_epsilon = svrf.error_min_epsilon(df_error, df_epsilon)
            df_gamma_cross_error_min_epsilon = svrf.gamma_cross_error_min_epsilon(
                df_gamma, df_error_min_epsilon
            )
            df_c_min_alpha_star = svrf.c_min_alpha_star(df_c, df_alpha_star)
            df_minus_alpha_star = svrf.convert_to_minus(df_alpha_star)
            df_max_alpha_star = svrf.data_maximum_alpha_star(
                df_gamma_cross_error_min_epsilon, df_minus_alpha_star
            )
            df_delta_alpha_star = svrf.data_minimum_alpha_star(
                df_max_alpha_star, df_c_min_alpha_star
            )

            df_minus_error = svrf.convert_to_minus(df_error)
            df_minus_alpha = svrf.convert_to_minus(df_alpha)
            df_c_min_alpha = svrf.c_min_alpha(df_c, df_alpha)
            df_min_error_min_epsilon = svrf.min_error_min_epsilon(
                df_minus_error, df_epsilon
            )
            df_gamma_cross_min_error_min_epsilon = svrf.gamma_cross_min_error_min_epsilon(
                df_gamma, df_min_error_min_epsilon
            )
            df_max_alpha = svrf.data_maximum_alpha(
                df_gamma_cross_min_error_min_epsilon, df_minus_alpha
            )
            df_delta_alpha = svrf.data_minimum_alpha(df_max_alpha, df_c_min_alpha)

            df_update_alpha = svrf.update_alpha(df_delta_alpha, df_alpha)
            df_update_alpha_star = svrf.update_alpha_star(
                df_delta_alpha_star, df_alpha_star
            )

            concat_df_train_iteration = [
                df_error,
                df_delta_alpha_star,
                df_delta_alpha,
                df_update_alpha_star,
                df_update_alpha,
            ]
            df_train_iteration = pd.concat(concat_df_train_iteration, axis=1)

            # Check condition
            ## max(abs(delta_alpha_star) < epsilon and max(abs(delta_alpha) < epsilon ==> Stop Iteration
            abs_delta_alpha_star = abs(df_delta_alpha_star)
            abs_delta_alpha = abs(df_delta_alpha)
            maximum_delta_alpha_star = (abs_delta_alpha_star.max()).max()
            maximum_delta_alpha = (abs_delta_alpha.max()).max()
            if (
                maximum_delta_alpha_star < epsilon and maximum_delta_alpha < epsilon
            ) and i < iteration:
                break

        df_updated_alpha_star = df_train_iteration[["update_alpha_star"]]
        df_updated_alpha = df_train_iteration[["update_alpha"]]
        df_updated_multipliers = svrf.alpha_star_min_alpha(
            df_updated_alpha_star, df_updated_alpha
        )

        df_multivarite = pd.read_csv(
            "dataset_mancanegara_kualanamu.csv", index_col="BulanTahun"
        )
        y_actual = df_multivarite[["DataAktual"]]

        df_distance_test = svrf.calcute_distance_test(X_test, X_train)
        df_kernel_test = calculate_kernel(df_distance_test)
        df_hessian_test = calculate_hessian(df_kernel_test)

        df_regression_function_test = svrf.regression_function(
            df_updated_multipliers, df_hessian_test
        )
        df_denormalized_y_test = svrf.denormalized_y_actual(y_test, y_actual)
        df_denormalized_y_pred_test = svrf.denormalized_y_pred(
            df_regression_function_test, y_actual
        )

        # df_prediction = pd.DataFrame(df_denormalized_y_pred_test.values, index=[y_test.index], columns=['Prediction'])
        df_predict_y_test_and_y_actual = pd.concat(
            [df_denormalized_y_test, df_denormalized_y_pred_test], axis=1
        )
        df_predict_y_test_and_y_actual = pd.DataFrame(
            df_predict_y_test_and_y_actual.values,
            index=[y_test.index],
            columns=["Data Aktual", "Prediksi"],
        )

        mape_test = svrf.calculate_mape(
            df_denormalized_y_test, df_denormalized_y_pred_test
        )

        return render_template(
            "predict-svr-multi.html",
            predictionMulti=df_predict_y_test_and_y_actual.to_html(
                classes="fixed-table"
            ),
            trainMulti=train_multi,
            testMulti=test_multi,
            clrMulti=cLR,
            cMulti=C,
            epsilonMulti=epsilon,
            lambdaMulti=_lambda,
            sigmaMulti=sigma,
            iterationMulti=iteration,
            mape_test=mape_test,
        )
    return render_template("predict-svr-multi.html")


@app.route("/about/")
def about():
    return render_template("about.html")


if __name__ == "__main___":
    app.run(debug=True)
