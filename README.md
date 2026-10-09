# Finansal Regresyon ve Altın Fiyatı Hesaplama Projesi

Bu proje, **Python** kullanılarak geliştirilmiş basit bir **finansal veri analizi ve regresyon** çalışmasıdır.  
Amaç, finansal veriler üzerinden **altın fiyatı tahmini / hesaplaması** yapmak ve regresyon yaklaşımını uygulamaktır.

Altının kapanış fiyatı (`gold close`), altının ve diğer piyasa göstergelerinin **geçmiş günlere ait (gecikmeli) değerleri** kullanılarak tahmin edilir ve üç regresyon modeli karşılaştırılır:

- Linear Regression
- Random Forest Regressor
- Decision Tree Regressor

---

## Veri Seti

Veri seti depoda `financial_regression.csv` adıyla bulunmaktadır.

- **3904** satır, **47** sütun
- Tarih aralığı: `2010-01-14` – `2024-10-23` (`date` sütunu)
- Sütunlar: S&P 500 (`sp500 *`), Nasdaq (`nasdaq *`), gümüş (`silver *`), petrol (`oil *`), platin (`platinum *`), paladyum (`palladium *`) ve altın (`gold *`) için open / high / low / close / volume / high-low değerleri; ayrıca `us_rates_%`, `CPI`, `usd_chf`, `eur_usd`, `GDP`.

---

## Yöntem

1. **Veri temizleme**
   - `date` sütunu çıkarılır.
   - Eksik değer oranı %30 ve üzeri olan sütunlar atılır.
   - Kalan eksik değerler önce ileri (`ffill`), sonra geri (`bfill`) doldurulur.
2. **Hedef ve öznitelikler**
   - Hedef: `gold close`
   - Veri sızıntısını (data leakage) önlemek için diğer tüm `gold *` sütunları çıkarılır.
   - Kullanılan piyasa göstergeleri: `oil close`, `silver close`, `platinum close`, `palladium close`, `sp500 close`, `nasdaq close`, `eur_usd`, `usd_chf`.
3. **Gecikmeli öznitelikler (lag features)**
   - `gold close` için 1, 2, 3, 5 ve 10 günlük gecikmeler
   - Seçili piyasa göstergeleri için 1, 2 ve 3 günlük gecikmeler
   - Kaydırma sonucu oluşan `NaN` satırlar atılır.
4. **Eğitim / test ayrımı**: Zaman sırası korunarak (`shuffle=False`) %80 eğitim / %20 test.
5. **Ölçekleme**: `StandardScaler` yalnızca Linear Regression için uygulanır.
6. **Modeller**
   - `LinearRegression()`
   - `RandomForestRegressor(n_estimators=400, random_state=42)`
   - `DecisionTreeRegressor(max_depth=12, random_state=42)`
7. **Değerlendirme**: MAE, MSE, RMSE ve R² metrikleri hesaplanıp R²'ye göre sıralı bir tablo olarak yazdırılır.
8. **Görselleştirme**
   - Test verisi üzerinde gerçek altın fiyatı ile model tahminlerinin karşılaştırma grafiği
   - Modellerin R² skorlarını gösteren çubuk grafik

---

## Sonuçlar

Depoda kaydedilmiş bir çıktı bulunmamaktadır. Metrik tablosunu ve grafikleri görmek için betiği çalıştırın.

---

## Proje Yapısı

```
.
├── altin_hesaplama.py          # Veri hazırlama, model eğitimi, değerlendirme ve grafikler
├── financial_regression.csv    # Veri seti
├── requirements.txt
└── README.md
```

---

## Çalıştırma

```bash
pip install -r requirements.txt
python altin_hesaplama.py
```

Betik `financial_regression.csv` dosyasını çalışma dizininde arar; bu nedenle komutu depo klasörü içinden çalıştırın.

---

## Gereksinimler

- Python 3
- pandas
- numpy
- matplotlib
- scikit-learn
