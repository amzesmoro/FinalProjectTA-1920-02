# Collection of Support Vector Regression (Univariate and Multivariate)

import pandas as pd
import math

# Re Frame to Supervised Forms (specially for univariate)
def reframe_to_supervised(data):
    target = ["DataAktual"]
    for i in range(1, 5):
        data["y_{}".format(i)] = data[target].shift(i)
    return data


## FUNCTION TO DATA TRAINING
# Calculation of Distance Data Train
# Formula: (xi-xj)^2
def calculate_distance_train(data_train):
    df_distance = [[] for i in range(len(data_train.index))]
    # i,j for index row data train
    # k for index column data train
    for i in range(len(data_train.index)):
        for j in range(len(data_train.index)):
            sum_row = 0
            distance = 0
            for k in range(len(data_train.columns)):
                distance = pow((data_train.values[i, k] - data_train.values[j, k]), 2)
                sum_row = sum_row + distance
            df_distance[j].append(sum_row)
    df_distance = pd.DataFrame(df_distance)
    return df_distance


# Initialization Multipliers Lagrange
## Alpha Star = 0 and Alpha = 0
def init_alpha(data):
    # i for index row data
    alpha = 0
    list_alpha = []
    for i in range(len(data.index)):
        list_alpha.append(alpha)
    df = pd.DataFrame(list_alpha, columns=["alpha"])
    return df


def init_alpha_star(data):
    # i for index row data
    alpha_star = 0
    list_alpha_star = []
    for i in range(len(data.index)):
        list_alpha_star.append(alpha_star)
    df = pd.DataFrame(list_alpha_star, columns=["alpha_star"])
    return df


def alpha_star_min_alpha(data_alpha_star, data_alpha):
    # i, j for index data alpha star
    # k, l for index data alpha
    k = 0
    l = 0
    list_alpha_star_min_alpha = []
    for i in range(len(data_alpha_star.index)):
        for j in range(len(data_alpha_star.columns)):
            sub = data_alpha_star.values[i, j] - data_alpha.values[k, l]
        k = k + 1
        list_alpha_star_min_alpha.append(sub)
    df = pd.DataFrame(list_alpha_star_min_alpha, columns=["alpha_star_min_alpha"])
    return df


# Calculation of Error
# Formula: E = yi - 𝝨(𝝰i*-𝝰i) * Rij
def multipliers_cross_hessian(data_multipliers, data_hessian):
    # i, j for index data multipliers
    # k, l for index data hessian
    df = [[] for i in range(len(data_multipliers.index))]
    for k in range(len(data_hessian.index)):
        sum_cross = 0
        for i in range(len(data_multipliers.index)):
            l = i
            for j in range(len(data_multipliers.columns)):
                cross = data_multipliers.values[i, j] * data_hessian.values[k, l]
                sum_cross = sum_cross + cross
        df[k].append(sum_cross)
    df = pd.DataFrame(df, columns=["multiplies_cross_hessian"])
    return df


def y_min_multipliers_cross_hessian(data_y, data_multipliers_cross_hessian):
    # i, j for index data y
    # k, l for index data multipliers cross hessian
    k = 0
    l = 0
    list_error = []
    for i in range(len(data_y.index)):
        for j in range(len(data_y.columns)):
            sub = data_y.values[i, j] - data_multipliers_cross_hessian.values[k, l]
        k = k + 1
        list_error.append(sub)
    df = pd.DataFrame(list_error, columns=["error"])
    return df


# Delta Lagrange Multipliers
## Fromula
## 𝝳𝝰i_star = min{max(𝝲(Ei-𝝴), -𝝰i_star), C-𝝰i_star}
## 𝝳𝝰i = min{max(𝝲(-Ei-𝝴), -𝝰i), C-𝝰i}
### Delta Alpha Star
# convert to dataframe
# episilon to dataframe
def epsilon_to_df(data_train, epsilon_value):
    list_epsilon = []
    for i in range(len(data_train.index)):
        list_epsilon.append(epsilon_value)
    df = pd.DataFrame(list_epsilon, columns=["epsilon"])
    return df


# gamma to dataframe
def gamma_to_df(data_train, gamma_value):
    list_gamma = []
    for i in range(len(data_train.index)):
        list_gamma.append(gamma_value)
    df = pd.DataFrame(list_gamma, columns=["gamma"])
    return df


# C to dataframe
def c_to_df(data_train, c_value):
    list_c = []
    for i in range(len(data_train.index)):
        list_c.append(c_value)
    df = pd.DataFrame(list_c, columns=["C"])
    return df


def error_min_epsilon(data_error, data_epsilon):
    # i, j for index data error
    # k, l for index data epsilon
    k = 0
    l = 0
    list_error_min_epsilon = []
    for i in range(len(data_error.index)):
        for j in range(len(data_error.columns)):
            sub = data_error.values[i, j] - data_epsilon.values[k, l]
        k = k + 1
        list_error_min_epsilon.append(sub)
    df = pd.DataFrame(list_error_min_epsilon, columns=["error_min_epsilon"])
    return df


def gamma_cross_error_min_epsilon(data_gamma, data_error_min_epsilon):
    # i, j for index data gamma
    # k, l for index data error min epsilon
    k = 0
    l = 0
    list_gamma_cross_error_min_epsilon = []
    for i in range(len(data_gamma.index)):
        for j in range(len(data_gamma.columns)):
            cross = data_gamma.values[i, j] * data_error_min_epsilon.values[k, l]
        k = k + 1
        list_gamma_cross_error_min_epsilon.append(cross)
    df = pd.DataFrame(
        list_gamma_cross_error_min_epsilon, columns=["gamma_cross_error_min_epsilon"]
    )
    return df


def c_min_alpha_star(data_c, data_alpha_star):
    # i, j for index data c
    # k, l for index data alpha star
    k = 0
    l = 0
    list_c_min_alpha_star = []
    for i in range(len(data_c.index)):
        for j in range(len(data_c.columns)):
            sub = data_c.values[i, j] - data_alpha_star.values[k, l]
        k = k + 1
        list_c_min_alpha_star.append(sub)
    df = pd.DataFrame(list_c_min_alpha_star, columns=["C_min_alpha_star"])
    return df


def convert_to_minus(data):
    df = data.apply(lambda x: x * -1)
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
            if data_1.values[i, j] > data_2.values[k, l]:
                maximum_value = data_1.values[i, j]
            else:
                maximum_value = data_2.values[k, l]
        k = k + 1
        list_max.append(maximum_value)
    df = pd.DataFrame(list_max, columns=["max_delta_alpha_star"])
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
            if data_1.values[i, j] > data_2.values[k, l]:
                maximum_value = data_1.values[i, j]
            else:
                maximum_value = data_2.values[k, l]
        k = k + 1
        list_max.append(maximum_value)
    df = pd.DataFrame(list_max, columns=["max_delta_alpha"])
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
            if data_1.values[i, j] < data_2.values[k, l]:
                minimum_value = data_1.values[i, j]
            else:
                minimum_value = data_2.values[k, l]
        k = k + 1
        list_min.append(minimum_value)
    df = pd.DataFrame(list_min, columns=["delta_alpha_star"])
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
            if data_1.values[i, j] < data_2.values[k, l]:
                minimum_value = data_1.values[i, j]
            else:
                minimum_value = data_2.values[k, l]
        k = k + 1
        list_min.append(minimum_value)
    df = pd.DataFrame(list_min, columns=["delta_alpha"])
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
            sub = data_c.values[i, j] - data_alpha.values[k, l]
        k = k + 1
        list_c_min_alpha.append(sub)
    df = pd.DataFrame(list_c_min_alpha, columns=["C_min_alpha"])
    return df


def min_error_min_epsilon(data_min_error, data_epsilon):
    # i, j for index data min error
    # k, l for index data epsilon
    k = 0
    l = 0
    list_min_error_min_epsilon = []
    for i in range(len(data_min_error.index)):
        for j in range(len(data_min_error.columns)):
            sub = data_min_error.values[i, j] - data_epsilon.values[k, l]
        k = k + 1
        list_min_error_min_epsilon.append(sub)
    df = pd.DataFrame(list_min_error_min_epsilon, columns=["min_error_min_epsilon"])
    return df


def gamma_cross_min_error_min_epsilon(data_gamma, data_min_error_min_epsilon):
    # i, j for index data gamma
    # k, l for index data min error min epsilon
    k = 0
    l = 0
    list_gamma_cross_min_error_min_epsilon = []
    for i in range(len(data_gamma.index)):
        for j in range(len(data_gamma.columns)):
            cross = data_gamma.values[i, j] * data_min_error_min_epsilon.values[k, l]
        k = k + 1
        list_gamma_cross_min_error_min_epsilon.append(cross)
    df = pd.DataFrame(
        list_gamma_cross_min_error_min_epsilon,
        columns=["gamma_cross_min_error_min_epsilon"],
    )
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
            update = data_delta_alpha_star.values[i, j] + data_alpha_star.values[k, l]
        k = k + 1
        list_update_alpha_star.append(update)
    df = pd.DataFrame(list_update_alpha_star, columns=["update_alpha_star"])
    return df


def update_alpha(data_delta_alpha, data_alpha):
    # i, j for index data delta alpha
    # k, l for index data alpha
    k = 0
    l = 0
    list_update_alpha = []
    for i in range(len(data_delta_alpha.index)):
        for j in range(len(data_delta_alpha.columns)):
            update = data_delta_alpha.values[i, j] + data_alpha.values[k, l]
        k = k + 1
        list_update_alpha.append(update)
    df = pd.DataFrame(list_update_alpha, columns=["update_alpha"])
    return df


# Regression function or y_pred
# f(x) = 𝝨(𝝰i_star-𝝰i)(K(xi,xj)+(𝝺^2)) or f(x) = 𝝨(𝝰i_star-𝝰i)Rij
def regression_function(data_updated_multipliers, data_hessian):
    # i, j for index data updated multipliers
    # k, l for index data hessian
    df = [[] for i in range(len(data_hessian.index))]
    for k in range(len(data_hessian.index)):
        sum_cross = 0
        for i in range(len(data_updated_multipliers.index)):
            l = i
            for j in range(len(data_updated_multipliers.columns)):
                cross = (
                    data_updated_multipliers.values[i, j] * data_hessian.values[k, l]
                )
                sum_cross = sum_cross + cross
        df[k].append(sum_cross)
    df = pd.DataFrame(df, columns=["f(x)"])
    return df


# Denormalized y and f(X)
# Formula: yi = Xn * (Xmax-Xmin) + Xmin
## Denormalized y
def denormalized_y_actual(data_to_denormalized, data_actual):
    # i, j for index data to denormalized
    list_data_to_denorm = []
    data_min_actual = min(data_actual.min())
    data_max_actual = max(data_actual.max())
    for i in range(len(data_to_denormalized.index)):
        for j in range(len(data_to_denormalized.columns)):
            data_denorm = (
                data_to_denormalized.values[i, j] * (data_max_actual - data_min_actual)
                + data_min_actual
            )
        list_data_to_denorm.append(data_denorm)
    df = pd.DataFrame(list_data_to_denorm, columns=["dernomalized_y"])
    return df


## Denormalized prediction
def denormalized_y_pred(data_to_denormalized, data_actual):
    # i, j for index data to denormalized
    list_data_to_denorm = []
    data_min_actual = min(data_actual.min())
    data_max_actual = max(data_actual.max())
    for i in range(len(data_to_denormalized.index)):
        for j in range(len(data_to_denormalized.columns)):
            data_denorm = (
                data_to_denormalized.values[i, j] * (data_max_actual - data_min_actual)
                + data_min_actual
            )
        list_data_to_denorm.append(data_denorm)
    df = pd.DataFrame(list_data_to_denorm, columns=["dernomalized_f(x)"])
    return df


# MAPE
# Formula: 1/n * 𝝨(|(yi-yi')/yi|) * 100%
def calculate_mape(data_1, data_2):
    # data_1 is denormalized y
    # data_2 is denormalized y_pred
    # i, j for index data 1
    # k, l for index data 2
    count_row = len(data_1)  # also same to len(data_2)
    sum_data = 0
    k = 0
    l = 0
    for i in range(len(data_1.index)):
        for j in range(len(data_1.columns)):
            sub_abs = (1 / count_row) * abs(
                (data_1.values[i, j] - data_2.values[k, l]) / data_1.values[i, j]
            )
            sum_data = sum_data + sub_abs
        k = k + 1
    mape = round(sum_data * 100, 2)
    return mape


## FUNCTION TO DATA TESTING
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
                distance = pow((data_test.values[i, j] - data_train.values[k, l]), 2)
                sum_row = sum_row + distance
            df[i].append(sum_row)
    df = pd.DataFrame(df)
    return df

