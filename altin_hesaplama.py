import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Veri Yükleme ve Temizleme

df = pd.read_csv("financial_regression.csv")

# Gereksiz tarih sütununu çıkar
df = df.drop(columns=["date"], errors="ignore")

# Çok fazla eksik değeri olan sütunları çıkar
missing_ratio = df.isnull().mean()
df = df.loc[:, missing_ratio < 0.3]

# Eksik değerleri ileri/geri doldurma
df = df.fillna(method="ffill").fillna(method="bfill")


# Hedef ve Öznitelikler

target = "gold close"

# Altınla ilgili diğer sütunları çıkar (data leakage önlemek için)
gold_cols = [c for c in df.columns if c.startswith("gold ") and c != target]
df = df.drop(columns=gold_cols)

# Diğer piyasa göstergeleri (daha küçük, anlamlı bir set)
predictors = [
    "oil close", "silver close", "platinum close", "palladium close",
    "sp500 close", "nasdaq close", "eur_usd", "usd_chf"
]
predictors = [c for c in predictors if c in df.columns]


# Özellik (Feature) Genişletme: Gecikmeli Değerler

fe = pd.DataFrame(index=df.index)
fe[target] = df[target]

# Altın fiyatının geçmiş 1,2,3,5,10 günlük değerlerini ekle
for L in [1, 2, 3, 5, 10]:
    fe[f"{target}_lag{L}"] = df[target].shift(L)

# Seçili piyasa göstergelerinin geçmiş 1,2,3 günlerini ekle
for c in predictors:
    for L in [1, 2, 3]:
        fe[f"{c}_lag{L}"] = df[c].shift(L)

# NaN satırlarını temizle
fe = fe.dropna().reset_index(drop=True)

X = fe.drop(columns=[target])
y = fe[target]


# 4. Train/Test Ayrımı (Zamana göre, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42
)


#  Ölçekleme (Sadece Linear Regression için)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeller

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=400, random_state=42, n_jobs=-1
    ),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42)
}

# Eğitim ve Değerlendirme

results = []
preds = {}

for name, model in models.items():
    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    preds[name] = y_pred
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results.append([name, mae, mse, rmse, r2])

# Sonuç tablosu
results_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R2"])
print("\n=== Model Performans Karşılaştırması ===\n")
print(results_df.sort_values("R2", ascending=False))


# Görselleştirme

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Gerçek (Gold Close)", color="black", linewidth=2)
for name, model in models.items():
    if name == "Linear Regression":
        plt.plot(model.predict(X_test_scaled), label=name)
    else:
        plt.plot(model.predict(X_test), label=name)
plt.title("Altın Fiyatı - Gerçek vs Tahmin")
plt.xlabel("Zaman (test verisi sırası)")
plt.ylabel("Fiyat")
plt.legend()
plt.grid(True)
plt.show()

# Bar grafiğiyle R² karşılaştırması
plt.figure(figsize=(8,5))
plt.bar(results_df["Model"], results_df["R2"], color=["#4285F4", "#34A853", "#FBBC05"])
plt.title("Model R² Karşılaştırması")
plt.ylabel("R² Skoru")
plt.grid(True, axis='y')
plt.show()
