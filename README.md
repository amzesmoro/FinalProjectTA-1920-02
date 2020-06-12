# Final Project TA1920-02
For final project purpose.
To create [flask project](https://code.visualstudio.com/docs/python/tutorial-flask)

## About
This prototype is an experimental result of Holt-Winters and Supports Vector Regression algorithm. This prototype will produce prediction results from the two algorithms based on the parameter values inputted by the user. Data that we are use in this experiment are Hotel Occupancy Rate (Tingkat Hunian Hotel), Events, Inflation (Inflasi), Exchange Rate USD to Rupiah and number of foreign tourists every month. The data we get from Badan Pusat Statistik Indonesia and Kementerian Pariwisata.

Holt Winters only uses one variable, namely the number of tourists, because the learning process only processes numerical data. While for Support Vector Regression is divided into two parts, namely Univariate SVR which also uses one variable, namely the number of tourists, and Multivariate SVR which uses several variables namely Hotel Occupancy Rate (Tingkat Hunian Hotel), Events, Inflation (Inflasi), Exchange Rate USD to Rupiah.

These two algorithms have different approaches. Holt Winters is used for time series data while Support Vector Regression is to see the effect of each variable on the target we want. So we can see the best algorithm to predict the number of tourists

------
If there are some improvemnt, feel free to contact us:
* Amzesmoro (amzesmoro05@gmail.com)
* Sandra Simangunsong

