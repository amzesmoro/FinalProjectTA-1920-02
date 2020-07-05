import os
from flask import Flask, flash, render_template, request, redirect, url_for, send_from_directory

import pandas as pd
import csv

from sklearn.preprocessing import MinMaxScaler
import math

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

### Holt Winters Functions

# 1. Menghitung Nilai Awal Pemodelan
# 1a. Menghitung Nilai Smoothing
def smoothingAwal(data):
    data_asli = data.iloc[:12,]
    S = []
    total = 0
    for x in data_asli:
        total = total + x
    smoothing = 1/12*(total)
    S.append(smoothing)
    return S

#Sl = 1/l(X1+X2+...+Xl)

# 1b. Menghitung Nilai Awal Musiman
def nilaiAwalMusimanMultiplicative(data, nilaiSmoothing):
    L = []
    data_asli = data.iloc[:12,]
    for x in data_asli:
        nilaiMusim = x/nilaiSmoothing[0]
        L.append(nilaiMusim)
    return L

#Lt = Xt/Sl

# 1b. Menghitung Nilai Awal Musiman
def nilaiAwalMusimanAdditive(data, nilaiSmoothing):
    L = []
    data_asli = data.iloc[:12,]
    for x in data_asli:
        nilaiMusim = x-nilaiSmoothing[0]
        L.append(nilaiMusim)
    return L

#Lt = Xt-Sl

# 1c. Menghitung Nilai Awal Tren
def nilaiTrenAwal(data):
    tren = []
    totalTren = 0
    data_asli = data.iloc[:12,]
    data_tren = data.iloc[12:24,]
    for x in range(12):
        total = (data_tren[12+x]-data_asli[x])/12
        totalTren = totalTren + total
    trenAwal = 1/12 * (totalTren)
    tren.append(trenAwal)
    return tren

#Bl= 1/12 (Xl+1-X1)/l + (Xl+2-X2)/l+...+(Xl+l-Xl)/l

# 2. Menghitung Nilai Smoothing Additive
# 2a. menghitung nilai Smoothing Keseluruhan
def smoothingKeseluruhanAdditive(alpa, data ,nilaiSmoothing, nilaiMusiman, nilaiTren, x):
    nilai_smooth =((alpa * (data[x+12] - nilaiMusiman[x])) + ((1-alpa) * (nilaiSmoothing[x] + nilaiTren[x])))
    return  nilai_smooth

#St = a(Xt-lt-l)+(1-a)(St-1+Bt-1)

# 2b. menghitung nilai tren
def trendSmoothingAdditive(beta, nilaiSmoothing, nilaiTren,x):
    nilai_tren_smooth = ((beta * (nilaiSmoothing[x+1]-nilaiSmoothing[x])) + ((1-beta)*nilaiTren[x]))
    return nilai_tren_smooth

#Bt = b(St-St-1)+(1-b)Bt-1

# 2c. menghitung nilai Musiman
def nilaiMusimanSmoothingAdditive(gamma, data, nilaiSmoothing, nilaiMusiman, x):
    nilai_musim = ((gamma * (data[x+12]-nilaiSmoothing[x+1])) + ((1-gamma)*(nilaiMusiman[x])))
    return nilai_musim

#lt = y(Xt-St)+(1-y)lt-L

# 3. Menghitung Nilai Smoothing Multipicative
# 3a. menghitung nilai Smoothing Keseluruhan
def smoothingKeseluruhanMultiplicative(alpa, data, nilaiSmoothing, nilaiMusiman, nilaiTren, x):
    nilai_smooth = ((alpa * (data[x+12] / nilaiMusiman[x])) + ((1-alpa) * (nilaiSmoothing[x] + nilaiTren[x])))
    return nilai_smooth

#St = a(Xt/lt-l)+(1-a)(St-1+Bt-1)

# 3b. menghitung nilai tren
def trendSmoothingMultiplicative(beta, nilaiSmoothing, nilaiTren, x):
    nilai_tren_smooth = ((beta * (nilaiSmoothing[x+1]-nilaiSmoothing[x])) + ((1-beta)*nilaiTren[x]))
    return nilai_tren_smooth

#Bt = b(St-St-1)+(1-b)Bt-1

# 3c. menghitung nilai Musiman
def nilaiMusimanSmoothingMultiplicative(gamma, data, nilaiSmoothing, nilaiMusiman, x):
    nilai_musim = ((gamma * (data[x+12] / nilaiSmoothing[x+1])) + ((1-gamma)*(nilaiMusiman[x])))
    return nilai_musim

#lt = y(Xt/St)+(1-y)lt-L

# 4. Menghitung Prediksi
# menghitung nilai Prediksi Additive
def nilaiPredictAdditive(nilaiSmoothing, nilaiTren, nilaiMusiman, x):
    nilai_predict = nilaiSmoothing[x] + nilaiTren[x] + nilaiMusiman[x]
    return nilai_predict

#Ft+m = St+Btm+lt-l+m

# menghitung nilai Prediksi Multipicative
def nilaiPredictMultiplicative(nilaiSmoothing, nilaiTren, nilaiMusiman, x):
    nilai_predict = ((nilaiSmoothing[x] + nilaiTren[x]) * (nilaiMusiman[x]))
    return nilai_predict

#Ft+m = (St+Btm)lt-l+m


### Support Vector Regression Function

## Reframe to supervised form
def reframe_to_supervised(data):    
    target = ['DataAktual']
    for i in range(1,5):
        data['y_{}'.format(i)] = data[target].shift(i)
    return data

## Step II
# Calculation of Distance Data Train    
def calculate_distance_train(data_train):
    df_distance = [[] for i in range(len(data_train.index))]
    # i,j for index row data train
    # k for index column data train
    for i in range(len(data_train.index)):  
        for j in range(len(data_train.index)):        
            sum_row = 0
            distance = 0
            for k in range(len(data_train.columns)):         
                distance = pow((data_train.values[i,k]-data_train.values[j,k]),2)
                sum_row = sum_row + distance
            df_distance[j].append(sum_row)
    df_distance = pd.DataFrame(df_distance)
    return df_distance

# Calculation of Kernel 

# Calculation of Matriks Hessian

## Step III - Step IV
# Initialization Multipliers Lagrange
## Alpha Star = 0 and Alpha = 0
def init_alpha(data):    
    # i for index row data
    alpha = 0
    list_alpha = []
    for i in range(len(data.index)):    
        list_alpha.append(alpha)
    df = pd.DataFrame(list_alpha, columns=['alpha'])    
    return df

def init_alpha_star(data):   
    # i for index row data
    alpha_star = 0
    list_alpha_star = []
    for i in range(len(data.index)):    
        list_alpha_star.append(alpha_star)
    df = pd.DataFrame(list_alpha_star, columns=['alpha_star'])    
    return df

def alpha_star_min_alpha(data_alpha_star, data_alpha):
    # i, j for index data alpha star
    # k, l for index data alpha 
    k = 0
    l = 0
    list_alpha_star_min_alpha = []
    for i in range(len(data_alpha_star.index)):
        for j in range(len(data_alpha_star.columns)):
            sub = data_alpha_star.values[i,j] - data_alpha.values[k,l]
        k = k + 1
        list_alpha_star_min_alpha.append(sub)
    df = pd.DataFrame(list_alpha_star_min_alpha, columns=['alpha_star_min_alpha'])     
    return df


# Calculation of Error
# Formula: E = yi - 𝝨(𝝰i*-𝝰i) * Rij
def multipliers_cross_hessian(data_multipliers,data_hessian):
    # i, j for index data multipliers
    # k, l for index data hessian
    df = [[] for i in range(len(data_multipliers.index))]
    for k in range(len(data_hessian.index)):
        sum_cross = 0
        for i in range(len(data_multipliers.index)):
            l = i
            for j in range(len(data_multipliers.columns)):
                cross = data_multipliers.values[i,j] * data_hessian.values[k,l]
                sum_cross = sum_cross + cross
        df[k].append(sum_cross)
    df = pd.DataFrame(df, columns=['multiplies_cross_hessian'])    
    return df

def y_min_multipliers_cross_hessian(data_y, data_multipliers_cross_hessian):
    # i, j for index data y
    # k, l for index data multipliers cross hessian
    k = 0
    l = 0
    list_error = []
    for i in range(len(data_y.index)):
        for j in range(len(data_y.columns)):
            sub = data_y.values[i,j] - data_multipliers_cross_hessian.values[k,l]
        k = k + 1
        list_error.append(sub)
    df = pd.DataFrame(list_error, columns=['error'])     
    return df


# Delta Lagrange Multipliers
## Fromula
## 𝝳𝝰i_star = min{max(𝝲(Ei-𝝴), -𝝰i_star), C-𝝰i_star}
## 𝝳𝝰i = min{max(𝝲(-Ei-𝝴), -𝝰i), C-𝝰i}
### Delta Alpha Star
# convert to dataframe
# episilon to dataframe
def epsilon_to_df(data_train,epsilon_value):
    list_epsilon = []
    for i in range(len(data_train.index)):
        list_epsilon.append(epsilon_value)
    df = pd.DataFrame(list_epsilon, columns=['epsilon'])    
    return df

# gamma to dataframe
def gamma_to_df(data_train,gamma_value):
    list_gamma = []
    for i in range(len(data_train.index)):
        list_gamma.append(gamma_value)
    df = pd.DataFrame(list_gamma, columns=['gamma'])    
    return df

# C to dataframe
def c_to_df(data_train,c_value):
    list_c = []
    for i in range(len(data_train.index)):
        list_c.append(c_value)
    df = pd.DataFrame(list_c, columns=['C'])    
    return df

def error_min_epsilon(data_error, data_epsilon):
    # i, j for index data error
    # k, l for index data epsilon
    k = 0
    l = 0
    list_error_min_epsilon = []
    for i in range(len(data_error.index)):
        for j in range(len(data_error.columns)):
            sub = data_error.values[i,j] - data_epsilon.values[k,l]
        k = k + 1
        list_error_min_epsilon.append(sub)
    df = pd.DataFrame(list_error_min_epsilon, columns=['error_min_epsilon'])     
    return df

def gamma_cross_error_min_epsilon(data_gamma, data_error_min_epsilon):
    # i, j for index data gamma
    # k, l for index data error min epsilon
    k = 0
    l = 0
    list_gamma_cross_error_min_epsilon = []
    for i in range(len(data_gamma.index)):
        for j in range(len(data_gamma.columns)):
            cross = data_gamma.values[i,j] * data_error_min_epsilon.values[k,l]
        k = k + 1
        list_gamma_cross_error_min_epsilon.append(cross)
    df = pd.DataFrame(list_gamma_cross_error_min_epsilon, columns=['gamma_cross_error_min_epsilon'])     
    return df

def c_min_alpha_star(data_c, data_alpha_star):
    # i, j for index data c
    # k, l for index data alpha star
    k = 0
    l = 0
    list_c_min_alpha_star = []
    for i in range(len(data_c.index)):
        for j in range(len(data_c.columns)):
            sub = data_c.values[i,j] - data_alpha_star.values[k,l]
        k = k + 1
        list_c_min_alpha_star.append(sub)
    df = pd.DataFrame(list_c_min_alpha_star, columns=['C_min_alpha_star'])     
    return df

def convert_to_minus(data):
    df = data.apply(lambda x:x*-1)
    return df
# max function for multipliers
## data maximum for alpha star
def data_maximum_alpha_star(data_1, data_2):
    # i, j for index data 1
    # k, l for index data 2
    k = 0
    l = 0
    list_max = []
    for i in range(len(data_1.index)):
        for j in range(len(data_1.columns)):  
            if(data_1.values[i,j] > data_2.values[k,l]):
                maximum_value = data_1.values[i,j]
            else:
                maximum_value = data_2.values[k,l]            
        k = k + 1
        list_max.append(maximum_value)
    df = pd.DataFrame(list_max, columns=['max_delta_alpha_star'])     
    return df

## data maximum for alpha
def data_maximum_alpha(data_1, data_2):
    # i, j for index data 1
    # k, l for index data 2
    k = 0
    l = 0
    list_max = []
    for i in range(len(data_1.index)):
        for j in range(len(data_1.columns)):  
            if(data_1.values[i,j] > data_2.values[k,l]):
                maximum_value = data_1.values[i,j]
            else:
                maximum_value = data_2.values[k,l]            
        k = k + 1
        list_max.append(maximum_value)
    df = pd.DataFrame(list_max, columns=['max_delta_alpha'])     
    return df

# min function for multipliers
## data minimum for alpha star
def data_minimum_alpha_star(data_1, data_2):
    # i, j for index data 1
    # k, l for index data 2
    k = 0
    l = 0
    list_min = []
    for i in range(len(data_1.index)):
        for j in range(len(data_1.columns)):  
            if(data_1.values[i,j] < data_2.values[k,l]):
                minimum_value = data_1.values[i,j]
            else:
                minimum_value = data_2.values[k,l]            
        k = k + 1
        list_min.append(minimum_value)
    df = pd.DataFrame(list_min, columns=['delta_alpha_star'])     
    return df

## data minimum for alpha
def data_minimum_alpha(data_1, data_2):
    # i, j for index data 1
    # k, l for index data 2
    k = 0
    l = 0
    list_min = []
    for i in range(len(data_1.index)):
        for j in range(len(data_1.columns)):  
            if(data_1.values[i,j] < data_2.values[k,l]):
                minimum_value = data_1.values[i,j]
            else:
                minimum_value = data_2.values[k,l]            
        k = k + 1
        list_min.append(minimum_value)
    df = pd.DataFrame(list_min, columns=['delta_alpha'])     
    return df

### Delta Alpha Star
def c_min_alpha(data_c, data_alpha):
    # i, j for index data c
    # k, l for index data alpha
    k = 0
    l = 0
    list_c_min_alpha = []
    for i in range(len(data_c.index)):
        for j in range(len(data_c.columns)):
            sub = data_c.values[i,j] - data_alpha.values[k,l]
        k = k + 1
        list_c_min_alpha.append(sub)
    df = pd.DataFrame(list_c_min_alpha, columns=['C_min_alpha'])     
    return df

def min_error_min_epsilon(data_min_error, data_epsilon):
    # i, j for index data min error
    # k, l for index data epsilon
    k = 0
    l = 0
    list_min_error_min_epsilon = []
    for i in range(len(data_min_error.index)):
        for j in range(len(data_min_error.columns)):
            sub = data_min_error.values[i,j] - data_epsilon.values[k,l]
        k = k + 1
        list_min_error_min_epsilon.append(sub)
    df = pd.DataFrame(list_min_error_min_epsilon, columns=['min_error_min_epsilon'])     
    return df

def gamma_cross_min_error_min_epsilon(data_gamma, data_min_error_min_epsilon):
    # i, j for index data gamma
    # k, l for index data min error min epsilon
    k = 0
    l = 0
    list_gamma_cross_min_error_min_epsilon = []
    for i in range(len(data_gamma.index)):
        for j in range(len(data_gamma.columns)):
            cross = data_gamma.values[i,j] * data_min_error_min_epsilon.values[k,l]
        k = k + 1
        list_gamma_cross_min_error_min_epsilon.append(cross)
    df = pd.DataFrame(list_gamma_cross_min_error_min_epsilon, columns=['gamma_cross_min_error_min_epsilon'])     
    return df

# New Lagrange Multipliers
## Formula:
## 𝝰i* (updated) = 𝝳𝝰i* + 𝝰i*
## 𝝰i (updated) = 𝝳𝝰i + 𝝰i
def update_alpha_star(data_delta_alpha_star, data_alpha_star):
    # i, j for index data delta alpha star
    # k, l for index data alpha star
    k = 0
    l = 0
    list_update_alpha_star = []
    for i in range(len(data_delta_alpha_star.index)):
        for j in range(len(data_delta_alpha_star.columns)):
            update = data_delta_alpha_star.values[i,j] + data_alpha_star.values[k,l]
        k = k + 1
        list_update_alpha_star.append(update)
    df = pd.DataFrame(list_update_alpha_star, columns=['update_alpha_star'])     
    return df

def update_alpha(data_delta_alpha, data_alpha):
    # i, j for index data delta alpha
    # k, l for index data alpha
    k = 0
    l = 0
    list_update_alpha = []
    for i in range(len(data_delta_alpha.index)):
        for j in range(len(data_delta_alpha.columns)):
            update = data_delta_alpha.values[i,j] + data_alpha.values[k,l]
        k = k + 1
        list_update_alpha.append(update)
    df = pd.DataFrame(list_update_alpha, columns=['update_alpha'])     
    return df

## Step V
# Regression function or y_pred            
def regression_function(data_updated_multipliers, data_hessian):
    # i, j for index data updated multipliers
    # k, l for index data hessian
    df = [[] for i in range(len(data_hessian.index))]
    for k in range(len(data_hessian.index)):
        sum_cross = 0
        for i in range(len(data_updated_multipliers.index)):
            l = i
            for j in range(len(data_updated_multipliers.columns)):
                cross = data_updated_multipliers.values[i,j] * data_hessian.values[k,l]
                sum_cross = sum_cross + cross
        df[k].append(sum_cross)
    df = pd.DataFrame(df, columns=['f(x)'])    
    return df

# Denormalized function
## Denormalized y
def denormalized_y_actual(data_to_denormalized, data_actual):
    # i, j for index data to denormalized
    list_data_to_denorm = []
    data_min_actual = min(data_actual.min())
    data_max_actual = max(data_actual.max())
    for i in range(len(data_to_denormalized.index)):
        for j in range(len(data_to_denormalized.columns)):
            data_denorm = data_to_denormalized.values[i,j] * (data_max_actual - data_min_actual) + data_min_actual
        list_data_to_denorm.append(data_denorm)
    df = pd.DataFrame(list_data_to_denorm, columns=['dernomalized_y'])    
    return df

## Denormalized prediction
def denormalized_y_pred(data_to_denormalized, data_actual):
    # i, j for index data to denormalized
    list_data_to_denorm = []
    data_min_actual = min(data_actual.min())
    data_max_actual = max(data_actual.max())
    for i in range(len(data_to_denormalized.index)):
        for j in range(len(data_to_denormalized.columns)):
            data_denorm = data_to_denormalized.values[i,j] * (data_max_actual - data_min_actual) + data_min_actual
        list_data_to_denorm.append(data_denorm)
    df = pd.DataFrame(list_data_to_denorm, columns=['Prediction f(x)'])    
    return df

# MAPE Train
def calculate_mape(data_1, data_2):
    # data_1 is denormalized y
    # data_2 is denormalized y_pred
    # i, j for index data 1
    # k, l for index data 2
    count_row = len(data_1) # also same to len(data_2)
    sum_data = 0
    k = 0
    l = 0
    for i in range(len(data_1.index)):
        for j in range(len(data_1.columns)):
            sub_abs = (1/count_row) * abs((data_1.values[i,j] - data_2.values[k,l]) / data_1.values[i,j])
            sum_data = sum_data + sub_abs
        k = k + 1
    mape = round(sum_data*100,2)    
    return mape        

# Data Test
# Calculation of Distance between Data Train and Data Test
def calcute_distance_test(data_test, data_train):
    # i, j for index data test
    # k, l for index data train
    df = [[] for i in range(len(data_test.index))]
    for i in range(len(data_test.index)):
        for k in range(len(data_train.index)):
            sum_row = 0
            distance = 0
            for j in range(len(data_test.columns)):
                l = j
                distance = pow((data_test.values[i,j] - data_train.values[k,l]),2)
                sum_row = sum_row + distance
            df[i].append(sum_row)
    df = pd.DataFrame(df)       
    return df            
 
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
        return render_template("holt-winters.html", 
            data = data_csv.to_html(classes = "fixed-table", header = False, index = False))  
    return render_template("holt-winters.html")

@app.route("/predict-holt-winters", methods=["GET", "POST"])
def predict_holt_winters():
    if request.method == "POST":        
        req = request.form
        alpa = float(req["alpha"])
        beta = float(req["beta"])
        gamma = float(req["gamma"])

        ## Additive
        df = pd.read_csv('data.csv', sep=',')
        data_additive = df.iloc[:,1]

        hasilPrediksiTrainingAdd=[]
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

        nilaiSmoothingAdditive = smoothingAwal(data_additive)    
        nilaiMusimanAdditive = nilaiAwalMusimanAdditive(data_additive, nilaiSmoothingAdditive)       
        nilaiTrenAdditive = nilaiTrenAwal(data_additive)
            
        data_pakai_additive = data_additive.iloc[12:,]
        predictAdditive=[]

        for x in range(0, len(data_pakai_additive)):
            nilai_smooth_additive = smoothingKeseluruhanAdditive(alpa, data_pakai_additive ,nilaiSmoothingAdditive, nilaiMusimanAdditive, nilaiTrenAdditive, x)
            nilaiSmoothingAdditive.append(nilai_smooth_additive)

            nilai_tren_additive = trendSmoothingAdditive(beta, nilaiSmoothingAdditive, nilaiTrenAdditive, x)
            nilaiTrenAdditive.append(nilai_tren_additive)

            nilai_musim_additive = nilaiMusimanSmoothingAdditive(gamma, data_pakai_additive, nilaiSmoothingAdditive, nilaiMusimanAdditive, x)
            nilaiMusimanAdditive.append(nilai_musim_additive)

            nilai_predict_additive = nilaiPredictAdditive(nilaiSmoothingAdditive, nilaiTrenAdditive, nilaiMusimanAdditive, x)
            predictAdditive.append(nilai_predict_additive)

        MAPEAdditive = 0
        errorAdditive = 0
            
        for i in range (0, len(data_pakai_additive)):
            totalAdditive = abs((predictAdditive[i] - data_pakai_additive[i+12])/predictAdditive[i]) 
            errorAdditive = errorAdditive + totalAdditive
                
        MAPEAdditive = errorAdditive / (len(data_pakai_additive))
        nilaiMAPEAdditive.append(MAPEAdditive)
            
        # MAPE Additive    
        MAPEAdditive = round(MAPEAdditive*100, 2)
            
        smoothingAdd.append(nilaiSmoothingAdditive)
        musimanAdd.append(nilaiMusimanAdditive)
        trenAdd.append(nilaiTrenAdditive)
        hasilPrediksiTrainingAdd.append(predictAdditive)

        smoothingadd = nilaiSmoothingAdditive[0]        

        musimanadd = nilaiMusimanAdditive[0:12]
        df_musimanadd = pd.DataFrame(musimanadd,
                    columns=["Nilai Musiman"],
                    index=['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 
                                'Agustus', 'September', 'Oktober', 'November', 'Desember'])

        trendadd = nilaiTrenAdditive[0]        
        
        MAPEterkecilAdd = nilaiMAPEAdditive[0]
        alpaFinalAdd = nilaiAlpa[0]
        betaFinalAdd = nilaiBeta[0]
        gammaFinalAdd = nilaiGamma[0]
        smoothFinalAdd = smoothingAdd[0]
        musimanFinalAdd = musimanAdd[0]
        trenFinalAdd = trenAdd[0]

        smoothAddFinal = [smoothFinalAdd[len(data_pakai_additive)]]
        musimanAddFinal = musimanFinalAdd[(len(data_pakai_additive)):]
        trenAddFinal = [trenFinalAdd[len(data_pakai_additive)]]
        prediksiAdd = []
        for x in range(12):
            pred = smoothAddFinal[x] + musimanAddFinal[x] + trenAddFinal[x]
            prediksiAdd.append(pred)
            smoothing = ((alpaFinalAdd*(prediksiAdd[x]-musimanAddFinal[x]))+ ((1-alpaFinalAdd)*(smoothAddFinal[x]-trenAddFinal[x])))
            smoothAddFinal.append(smoothing)
            trensmoothing = ((betaFinalAdd*((smoothAddFinal[x+1])-(smoothAddFinal[x]))) + ((1-betaFinalAdd)*trenAddFinal[x]))
            trenAddFinal.append(trensmoothing) 
            musim = (gammaFinalAdd*(prediksiAdd[x]-smoothAddFinal[x+1])) + ((1-gammaFinalAdd)*musimanAddFinal[x])
            musimanAddFinal.append(musim)
        
        # Prediction Additive
        data_additive = prediksiAdd
        df_prediction_add = pd.DataFrame(data_additive, 
                        columns=["Prediksi Wisatawan"],
                        index=['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 
                                'Agustus', 'September', 'Oktober', 'November', 'Desember'])

        ## Multiplicative
        data_multiplicative = df.iloc[:,1]

        hasilPrediksiTrainingMul=[]
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

        nilaiSmoothingMultiplicative = smoothingAwal(data_multiplicative)
        nilaiMusimanMultiplicative = nilaiAwalMusimanMultiplicative(data_multiplicative, nilaiSmoothingMultiplicative)
        nilaiTrenMultiplicative = nilaiTrenAwal(data_multiplicative)

        data_pakai_multiplicative = data_multiplicative.iloc[12:,]
        predictMultiplicative=[]

        for x in range(0, len(data_pakai_multiplicative)):
            nilai_smooth_multiplicative = smoothingKeseluruhanMultiplicative(alpa, data_pakai_multiplicative ,nilaiSmoothingMultiplicative, nilaiMusimanMultiplicative, nilaiTrenMultiplicative, x)
            nilaiSmoothingMultiplicative.append(nilai_smooth_multiplicative)

            nilai_tren_multiplicative = trendSmoothingMultiplicative(beta, nilaiSmoothingMultiplicative, nilaiTrenMultiplicative, x)
            nilaiTrenMultiplicative.append(nilai_tren_multiplicative)

            nilai_musim_multiplicative = nilaiMusimanSmoothingMultiplicative(gamma, data_pakai_multiplicative, nilaiSmoothingMultiplicative, nilaiMusimanMultiplicative, x)
            nilaiMusimanMultiplicative.append(nilai_musim_multiplicative)

            nilai_predict_multiplicative = nilaiPredictMultiplicative(nilaiSmoothingMultiplicative, nilaiTrenMultiplicative, nilaiMusimanMultiplicative, x)
            predictMultiplicative.append(nilai_predict_multiplicative)
            
        MAPEMultiplicative = 0
        errorMultiplicative = 0
            
        for i in range (0, len(data_pakai_multiplicative)):
            totalMultiplicative = abs((predictMultiplicative[i] - data_pakai_multiplicative[i+12])/predictMultiplicative[i])
            errorMultiplicative = errorMultiplicative + totalMultiplicative
            
        MAPEMultiplicative = errorMultiplicative / (len(data_pakai_multiplicative))
        nilaiMAPEMultiplicative.append(MAPEMultiplicative)

        # MAPE Multiplicative  
        MAPEMultiplicative = round(MAPEMultiplicative*100, 2)
            
        smoothingMul.append(nilaiSmoothingMultiplicative)
        musimanMul.append(nilaiMusimanMultiplicative)
        trenMul.append(nilaiTrenMultiplicative)
        hasilPrediksiTrainingMul.append(predictMultiplicative)

        smoothingmul = nilaiSmoothingMultiplicative[0]

        musimanmul = nilaiMusimanMultiplicative[0:12]
        df_musimanmul = pd.DataFrame(musimanmul,
                    columns=["Nilai Musiman"],
                    index=['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 
                                'Agustus', 'September', 'Oktober', 'November', 'Desember'])
        
        trendmul = nilaiTrenMultiplicative[0]
        
        MAPEterkecilMul = nilaiMAPEMultiplicative[0]
        alpaFinalMul = nilaiAlpa[0]
        betaFinalMul = nilaiBeta[0]
        gammaFinalMul = nilaiGamma[0]
        smoothFinalMul = smoothingMul[0]
        musimanFinalMul = musimanMul[0]
        trenFinalMul = trenMul[0]

        smoothMulFinal = [smoothFinalMul[len(data_pakai_multiplicative)]]
        musimanMulFinal = musimanFinalMul[(len(data_pakai_multiplicative)):]
        trenMulFinal = [trenFinalMul[len(data_pakai_multiplicative)]]
        prediksiMul = []
        for x in range(12):
            predik = (smoothMulFinal[x] + trenMulFinal[x])*musimanMulFinal[x]
            prediksiMul.append(predik)
            smoothing = ((alpaFinalMul * (prediksiMul[x]/musimanMulFinal[x])) + ((1-alpaFinalMul) * (smoothMulFinal[x] + trenMulFinal[x])))
            smoothMulFinal.append(smoothing)
            trensmoothing = ((betaFinalMul*((smoothMulFinal[x+1])-(smoothMulFinal[x]))) + ((1-betaFinalMul)*trenMulFinal[x]))
            trenMulFinal.append(trensmoothing)
            musim = (gammaFinalMul*(prediksiMul[x]/smoothMulFinal[x+1])) + ((1-gammaFinalMul)*musimanMulFinal[x])
            musimanMulFinal.append(musim)

        data_multiplicative = prediksiMul
        df_prediction_mul = pd.DataFrame(data_multiplicative,
                        columns=["Prediksi Wisatawan"],
                        index=['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 
                                'Agustus', 'September', 'Oktober', 'November', 'Desember'])

        return render_template("predict-holt-winters.html", 
            alpha = alpa, beta = beta, gamma = gamma,
            mape_additive = MAPEAdditive, 
            prediction_additive = df_prediction_add.to_html(classes = "fixed-table"), 
            mape_multiplicative = MAPEMultiplicative, 
            prediction_multiplicative = df_prediction_mul.to_html(classes = "fixed-table"),
            smoothingadd = round(smoothingadd, 2), trendadd = round(trendadd, 2), 
            smoothingmul = round(smoothingmul, 2), trendmul = round(trendmul, 2),
            df_musimanadd = df_musimanadd.to_html(classes = "fixed-table"),
            df_musimanmul = df_musimanmul.to_html(classes = "fixed-table"))
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

        dataset = pd.read_csv('dataset_mancanegara_kualanamu.csv', index_col='BulanTahun')
        dataset = dataset[['DataAktual']]

        scaler = MinMaxScaler(feature_range=(0,1))
        col_to_norm = ['DataAktual']
        dataset[col_to_norm] = scaler.fit_transform(dataset[col_to_norm])

        reorder_cols = ['y_4', 'y_3', 'y_2', 'y_1', 'DataAktual']
        df_reframe = reframe_to_supervised(dataset)
        df_reframe = df_reframe.reindex(columns=reorder_cols)
        dataset_univariate = df_reframe.dropna()

        return render_template("svr-uni.html", 
            data = data.to_html(classes="fixed-table", header=False, index=False), 
            data_scaled = dataset[col_to_norm].to_html(classes="fixed-table"), 
            data_reframed = df_reframe.dropna().to_html(classes="fixed-table"))  
    return render_template("svr-uni.html")

@app.route("/predict-svr-uni", methods=["GET", "POST"])
def predict_svr_uni():
    if request.method == "POST":
        dataset = pd.read_csv('dataset_mancanegara_kualanamu.csv', index_col='BulanTahun')
        dataset = dataset[['DataAktual']]

        scaler = MinMaxScaler(feature_range=(0,1))
        col_to_norm = ['DataAktual']
        dataset[col_to_norm] = scaler.fit_transform(dataset[col_to_norm])

        reorder_cols = ['y_4', 'y_3', 'y_2', 'y_1', 'DataAktual']
        df_reframe = reframe_to_supervised(dataset)
        df_reframe = df_reframe.reindex(columns=reorder_cols)
        dataset_univariate = df_reframe.dropna()

        dataset_univariate.columns = ['y_4', 'y_3', 'y_2', 'y_1', 'y']

        features = ['y_4', 'y_3', 'y_2', 'y_1']
        target = ['y']
        X = dataset_univariate[features]
        y = dataset_univariate[target]

        # Splitting Data
        train_uni = int(request.form["train-uni"])
        test_uni = int(request.form["test-uni"])
        
        split_dataset = int((train_uni/100)*len(X))

        X_train, X_test = X[:split_dataset], X[split_dataset:]
        y_train, y_test = y[:split_dataset], y[split_dataset:]

        cLR = float(request.form["clr-uni"])
        C = float(request.form["c-uni"])
        epsilon = float(request.form["epsilon-uni"])
        _lambda = float(request.form["lambda-uni"])
        sigma = float(request.form["sigma-uni"])
        iteration = int(request.form["iteration-uni"])

        df_distance_train = calculate_distance_train(X_train)

        def calculate_kernel(data):            
            # i, j index for data
            df_kernel = [[] for i in range(len(data.index))]
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    kernel = math.exp(-(data.values[i,j])/(2*pow(sigma, 2)))
                    df_kernel[i].append(kernel)
            df_kernel = pd.DataFrame(df_kernel)       
            return df_kernel  

        df_kernel_train = calculate_kernel(df_distance_train)

        def calculate_hessian(data):            
            # i, j index for data
            df_hessian = [[] for i in range(len(data.index))]
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    hessian = data.values[i,j] + pow(_lambda,2)
                    df_hessian[i].append(hessian)
            df_hessian = pd.DataFrame(df_hessian)       
            return df_hessian

        df_hessian_train = calculate_hessian(df_kernel_train)

        gamma = round(cLR/max(df_hessian_train.max()),3)

        for i in range(iteration):    
            data_train = y_train
            data_hessian = df_hessian_train
            
            if i == 0:
                df_alpha = init_alpha(data_train)
                df_alpha_star = init_alpha_star(data_train)
            elif i > 0:
                df_alpha = df_update_alpha
                df_alpha_star = df_update_alpha_star
            
            df_multipliers = alpha_star_min_alpha(df_alpha_star, df_alpha)
            df_multipliers_cross_hessian = multipliers_cross_hessian(df_multipliers, data_hessian)
            df_error = y_min_multipliers_cross_hessian(data_train, df_multipliers_cross_hessian)
            
            df_epsilon = epsilon_to_df(data_train, epsilon)
            df_gamma = gamma_to_df(data_train, gamma)
            df_c = c_to_df(data_train, C)
            
            df_error_min_epsilon = error_min_epsilon(df_error, df_epsilon)
            df_gamma_cross_error_min_epsilon = gamma_cross_error_min_epsilon(df_gamma, df_error_min_epsilon)
            df_c_min_alpha_star = c_min_alpha_star(df_c, df_alpha_star)
            df_minus_alpha_star = convert_to_minus(df_alpha_star)
            df_max_alpha_star = data_maximum_alpha_star(df_gamma_cross_error_min_epsilon, df_minus_alpha_star)
            df_delta_alpha_star = data_minimum_alpha_star(df_max_alpha_star, df_c_min_alpha_star)
            
            df_minus_error = convert_to_minus(df_error)
            df_minus_alpha = convert_to_minus(df_alpha)
            df_c_min_alpha = c_min_alpha(df_c, df_alpha)
            df_min_error_min_epsilon = min_error_min_epsilon(df_minus_error, df_epsilon)
            df_gamma_cross_min_error_min_epsilon = gamma_cross_min_error_min_epsilon(df_gamma, df_min_error_min_epsilon)
            df_max_alpha = data_maximum_alpha(df_gamma_cross_min_error_min_epsilon, df_minus_alpha)
            df_delta_alpha = data_minimum_alpha(df_max_alpha, df_c_min_alpha)
            
            df_update_alpha = update_alpha(df_delta_alpha, df_alpha)
            df_update_alpha_star = update_alpha_star(df_delta_alpha_star, df_alpha_star)
            
            concat_df_train_iteration = [df_error, df_delta_alpha_star, df_delta_alpha, df_update_alpha_star, df_update_alpha]    
            df_train_iteration = pd.concat(concat_df_train_iteration, axis=1)
                    
        df_updated_alpha_star = df_train_iteration[['update_alpha_star']]
        df_updated_alpha  = df_train_iteration[['update_alpha']]

        df_updated_multipliers = alpha_star_min_alpha(df_updated_alpha_star, df_updated_alpha)

        df_regression_function_train = regression_function(df_updated_multipliers, df_hessian_train)

        # Denormalized y and f(X)
        df_univarite = pd.read_csv('dataset_mancanegara_kualanamu.csv', index_col='BulanTahun')
        y_actual = df_univarite[['DataAktual']]

        df_denormalized_y_train = denormalized_y_actual(y_train, y_actual)

        df_denormalized_y_pred_train = denormalized_y_pred(df_regression_function_train, y_actual)

        mape_train = calculate_mape(df_denormalized_y_train, df_denormalized_y_pred_train)

        df_distance_test = calcute_distance_test(X_test, X_train)

        df_kernel_test = calculate_kernel(df_distance_test)

        df_hessian_test = calculate_hessian(df_kernel_test)

        df_regression_function_test = regression_function(df_updated_multipliers, df_hessian_test)

        df_denormalized_y_test = denormalized_y_actual(y_test, y_actual)

        df_denormalized_y_pred_test = denormalized_y_pred(df_regression_function_test, y_actual)

        mape_test = calculate_mape(df_denormalized_y_test, df_denormalized_y_pred_test)

        return render_template("predict-svr-uni.html", 
            predictionUni = df_denormalized_y_pred_test.to_html(classes="fixed-table"), 
            trainUni = train_uni, testUni = test_uni, clrUni = cLR, 
            cUni = C, epsilonUni = epsilon, lambdaUni = _lambda, sigmaUni = sigma, 
            iterationUni = iteration, mape_test = mape_test)  
    return render_template("predict-svr-uni.html")   

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

        dataset_multivariate = pd.read_csv('dataset_mancanegara_kualanamu.csv', index_col='BulanTahun')

        scaler = MinMaxScaler(feature_range=(0,1))

        features_to_norm = ['TingkatHunianHotel(%)','Events','Inflasi','USDToRupiah','DataAktual']
        dataset_multivariate[features_to_norm] = scaler.fit_transform(dataset_multivariate[features_to_norm])

        dataset_multivariate.columns = ['X1', 'X2', 'X3', 'X4', 'y']

        return render_template("svr-multi.html", 
            data = data.to_html(classes="fixed-table", header=False, index=False),
            data_scaled = dataset_multivariate.to_html(classes="fixed-table")) 
    return render_template("svr-multi.html")

@app.route("/predict-svr-multi", methods=["GET", "POST"])
def predict_svr_multi():
    if request.method == "POST":       

        dataset_multivariate = pd.read_csv('dataset_mancanegara_kualanamu.csv', index_col='BulanTahun')

        scaler = MinMaxScaler(feature_range=(0,1))

        features_to_norm = ['TingkatHunianHotel(%)','Events','Inflasi','USDToRupiah','DataAktual']
        dataset_multivariate[features_to_norm] = scaler.fit_transform(dataset_multivariate[features_to_norm])

        dataset_multivariate.columns = ['X1', 'X2', 'X3', 'X4', 'y']

        features = ['X1', 'X2', 'X3', 'X4']
        target = ['y']
        X = dataset_multivariate[features]
        y = dataset_multivariate[target]

        # Splitting Data
        train_multi = int(request.form["train-multi"])
        test_multi = int(request.form["test-multi"])

        split_dataset = int((train_multi/100)*len(X))
        X_train, X_test = X[:split_dataset], X[split_dataset:]
        y_train, y_test = y[:split_dataset], y[split_dataset:]

        cLR = float(request.form["clr-multi"])
        C = float(request.form["c-multi"])
        epsilon = float(request.form["epsilon-multi"])
        _lambda = float(request.form["lambda-multi"])
        sigma = float(request.form["sigma-multi"])
        iteration = int(request.form["iteration-multi"])

        df_distance_train = calculate_distance_train(X_train)

        def calculate_kernel(data):
            # i, j index for data
            df_kernel = [[] for i in range(len(data.index))]
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    kernel = math.exp(-(data.values[i,j])/(2*pow(sigma,2)))
                    df_kernel[i].append(kernel)
            df_kernel = pd.DataFrame(df_kernel)       
            return df_kernel  

        df_kernel_train = calculate_kernel(df_distance_train)    

        def calculate_hessian(data):
            # i, j index for data
            df_hessian = [[] for i in range(len(data.index))]
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    hessian = data.values[i,j] + pow(_lambda,2)
                    df_hessian[i].append(hessian)
            df_hessian = pd.DataFrame(df_hessian)       
            return df_hessian

        df_hessian_train = calculate_hessian(df_kernel_train)

        gamma = round(cLR/max(df_hessian_train.max()),3)

        for i in range(iteration):   
            data_train = y_train
            data_hessian = df_hessian_train
            
            if i == 0:
                df_alpha = init_alpha(data_train)
                df_alpha_star = init_alpha_star(data_train)
            elif i > 0:
                df_alpha = df_update_alpha
                df_alpha_star = df_update_alpha_star
            
            df_multipliers = alpha_star_min_alpha(df_alpha_star, df_alpha)
            df_multipliers_cross_hessian = multipliers_cross_hessian(df_multipliers, data_hessian)
            df_error = y_min_multipliers_cross_hessian(data_train, df_multipliers_cross_hessian)
            
            df_epsilon = epsilon_to_df(data_train, epsilon)
            df_gamma = gamma_to_df(data_train, gamma)
            df_c = c_to_df(data_train, C)
            
            df_error_min_epsilon = error_min_epsilon(df_error, df_epsilon)
            df_gamma_cross_error_min_epsilon = gamma_cross_error_min_epsilon(df_gamma, df_error_min_epsilon)
            df_c_min_alpha_star = c_min_alpha_star(df_c, df_alpha_star)
            df_minus_alpha_star = convert_to_minus(df_alpha_star)
            df_max_alpha_star = data_maximum_alpha_star(df_gamma_cross_error_min_epsilon, df_minus_alpha_star)
            df_delta_alpha_star = data_minimum_alpha_star(df_max_alpha_star, df_c_min_alpha_star)
            
            df_minus_error = convert_to_minus(df_error)
            df_minus_alpha = convert_to_minus(df_alpha)
            df_c_min_alpha = c_min_alpha(df_c, df_alpha)
            df_min_error_min_epsilon = min_error_min_epsilon(df_minus_error, df_epsilon)
            df_gamma_cross_min_error_min_epsilon = gamma_cross_min_error_min_epsilon(df_gamma, df_min_error_min_epsilon)
            df_max_alpha = data_maximum_alpha(df_gamma_cross_min_error_min_epsilon, df_minus_alpha)
            df_delta_alpha = data_minimum_alpha(df_max_alpha, df_c_min_alpha)
            
            df_update_alpha = update_alpha(df_delta_alpha, df_alpha)
            df_update_alpha_star = update_alpha_star(df_delta_alpha_star, df_alpha_star)
            
            concat_df_train_iteration = [df_error, df_delta_alpha_star, df_delta_alpha, df_update_alpha_star, df_update_alpha]    
            df_train_iteration = pd.concat(concat_df_train_iteration, axis=1)


        df_updated_alpha_star = df_train_iteration[['update_alpha_star']]
        df_updated_alpha  = df_train_iteration[['update_alpha']]

        df_updated_multipliers = alpha_star_min_alpha(df_updated_alpha_star, df_updated_alpha)

        df_regression_function_train = regression_function(df_updated_multipliers, df_hessian_train)

        df_multivarite = pd.read_csv('dataset_mancanegara_kualanamu.csv', index_col='BulanTahun')
        y_actual = df_multivarite[['DataAktual']]

        df_denormalized_y_train = denormalized_y_actual(y_train, y_actual)

        df_denormalized_y_pred_train = denormalized_y_pred(df_regression_function_train, y_actual)

        mape_train = calculate_mape(df_denormalized_y_train, df_denormalized_y_pred_train)

        df_distance_test = calcute_distance_test(X_test, X_train)

        df_kernel_test = calculate_kernel(df_distance_test)

        df_hessian_test = calculate_hessian(df_kernel_test)

        df_regression_function_test = regression_function(df_updated_multipliers, df_hessian_test)

        df_denormalized_y_test = denormalized_y_actual(y_test, y_actual)

        df_denormalized_y_pred_test = denormalized_y_pred(df_regression_function_test, y_actual)

        mape_test = calculate_mape(df_denormalized_y_test, df_denormalized_y_pred_test)

        return render_template("predict-svr-multi.html",
            predictionMulti = df_denormalized_y_pred_test.to_html(classes="fixed-table"),
            trainMulti = train_multi, testMulti = test_multi, clrMulti = cLR, 
            cMulti = C, epsilonMulti = epsilon, lambdaMulti = _lambda, sigmaMulti = sigma, 
            iterationMulti = iteration, mape_test = mape_test) 
    return render_template("predict-svr-multi.html")

@app.route("/about/")
def about():
    return render_template("about.html")

if __name__ == "__main___":
    app.run(debug=True)