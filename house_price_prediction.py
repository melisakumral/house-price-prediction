'''import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('train.csv')
import pandas as pd

# Veriyi yükle
data = pd.read_csv('train.csv')

# İlk 5 satırı gör
print("İlk 5 satır:")
print(data.head())

# Veri tipi ve eksik değer var mı ona bak
print("\nVeri hakkında genel bilgi:")
print(data.info())

# Sayısal sütunların temel istatistikleri
print("\nSayısal sütunların istatistikleri:")
print(data.describe())

# Her sütunda kaç tane eksik (NaN) değer var kontrol et
print("\nEksik değer sayısı sütun bazında:")
print(data.isnull().sum())

missing_values = data.isnull().sum()
print("Eksik veri olan sütunlar:")
print(missing_values[missing_values > 0])
# Sayısal sütunlar
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    median = data[col].median()
    data[col].fillna(median, inplace=True)
    print(f"{col} sütunundaki eksikler medyan ({median}) ile dolduruldu.")

# Kategorik sütunlar
cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    mode = data[col].mode()[0]
    data[col].fillna(mode, inplace=True)
    print(f"{col} sütunundaki eksikler mod ({mode}) ile dolduruldu.")
data = pd.get_dummies(data, drop_first=True)
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R2 Skoru: {r2:.2f}")

plt.figure(figsize=(10,6))
sns.histplot(data['SalePrice'], bins=50, kde=True)
plt.title('Satış Fiyatı Dağılımı')
plt.xlabel('Satış Fiyatı')
plt.ylabel('Frekans')
plt.show()
corr_matrix = data.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, cmap='coolwarm', square=True)
plt.title('Değişkenler Arası Korelasyon Isı Haritası')
plt.show()
plt.figure(figsize=(8,6))
sns.scatterplot(x=data['GrLivArea'], y=data['SalePrice'])
plt.title('Yaşanabilir Alan (GrLivArea) ile Satış Fiyatı')
plt.xlabel('Yaşanabilir Alan (sq ft)')
plt.ylabel('Satış Fiyatı')
plt.show()
'''
'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Veriyi yükle
data = pd.read_csv('train.csv')

# 2. Eksik değerleri doldur
# Sayısal sütunlar için medyan ile doldurma
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    median = data[col].median()
    data[col] = data[col].fillna(median)  # inplace yerine atama yapıyoruz

# Kategorik sütunlar için mod ile doldurma
cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    mode = data[col].mode()[0]
    data[col] = data[col].fillna(mode)  # inplace yerine atama yapıyoruz

# 3. Kategorik verileri sayısallaştır (one-hot encoding)
data = pd.get_dummies(data, drop_first=True)

# 4. Özellik ve hedef değişkenleri ayır
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# 5. Veriyi eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 6. Decision Tree modeli oluştur ve eğit
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# 7. Tahmin yap
y_pred = model.predict(X_test)

# 8. Performans ölçümü yap
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Decision Tree RMSE: {rmse:.2f}")
print(f"Decision Tree R2: {r2:.2f}")

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Veri yükleme ve ön işleme (aynı kod)
data = pd.read_csv('train.csv')

num_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    median = data[col].median()
    data[col] = data[col].fillna(median)

cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    mode = data[col].mode()[0]
    data[col] = data[col].fillna(mode)

data = pd.get_dummies(data, drop_first=True)

X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Modeller
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Performans ölçümü fonksiyonu
def print_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} RMSE: {rmse:.2f}")
    print(f"{model_name} R2: {r2:.2f}\n")

print_metrics(y_test, y_pred_lr, "Linear Regression")
print_metrics(y_test, y_pred_dt, "Decision Tree Regressor")

# Grafik çizimi

plt.figure(figsize=(14, 6))

# 1. Gerçek Değerler vs Linear Regression Tahmini
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.5, color='blue', label='LR Tahmini')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Satış Fiyatı')
plt.ylabel('Tahmin Edilen Satış Fiyatı')
plt.title('Linear Regression: Gerçek vs Tahmin')
plt.legend()

# 2. Gerçek Değerler vs Decision Tree Tahmini
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_dt, alpha=0.5, color='green', label='DT Tahmini')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Satış Fiyatı')
plt.ylabel('Tahmin Edilen Satış Fiyatı')
plt.title('Decision Tree: Gerçek vs Tahmin')
plt.legend()

plt.tight_layout()
plt.show()

