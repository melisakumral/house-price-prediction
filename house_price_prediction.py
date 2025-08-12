import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ========================
# Veri Yükleme ve Ön İşleme
# ========================

data = pd.read_csv('train.csv')

# Sayısal sütunlardaki eksik değerleri medyan ile doldurma
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    median = data[col].median()
    data[col] = data[col].fillna(median)

# Kategorik sütunlardaki eksik değerleri mod ile doldurma
cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    mode = data[col].mode()[0]
    data[col] = data[col].fillna(mode)

# Kategorik değişkenleri one-hot encoding ile sayısala çevirme
data = pd.get_dummies(data, drop_first=True)

# Özellikler ve hedef değişkenin ayrılması
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Eğitim ve test verisine bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ========================
# 1. Doğrusal Regresyon Modeli
# ========================

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# ========================
# 2. Karar Ağacı (Decision Tree) Modeli
# ========================

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# ========================
# Performans Ölçümü Fonksiyonu
# ========================

def print_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} RMSE: {rmse:.2f}")
    print(f"{model_name} R2: {r2:.2f}\n")

print_metrics(y_test, y_pred_lr, "Linear Regression")
print_metrics(y_test, y_pred_dt, "Decision Tree Regressor")

# ========================
# Grafiklerle Model Karşılaştırması
# ========================

plt.figure(figsize=(14, 6))

# Linear Regression: Gerçek ve Tahmin Edilen Değerler
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.5, color='blue', label='LR Tahmini')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Satış Fiyatı')
plt.ylabel('Tahmin Edilen Satış Fiyatı')
plt.title('Linear Regression: Gerçek vs Tahmin')
plt.legend()

# Decision Tree: Gerçek ve Tahmin Edilen Değerler
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_dt, alpha=0.5, color='green', label='DT Tahmini')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Satış Fiyatı')
plt.ylabel('Tahmin Edilen Satış Fiyatı')
plt.title('Decision Tree: Gerçek vs Tahmin')
plt.legend()

plt.tight_layout()
plt.show()
