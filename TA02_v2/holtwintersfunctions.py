# Collection of Holt-Winters

## Inialisasi Nilai Awal, Smoothing & Prediksi
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
