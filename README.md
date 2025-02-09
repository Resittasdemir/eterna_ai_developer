# 📊 Flask Tahmin API'si

Bu proje, zaman serisi tahmini yapmak için geliştirilen bir **Flask API**'sidir. **Makine öğrenimi modelleri** kullanarak belirli tarihler için tahminlerde bulunur. Projede **Rate Limiting**, **Hyperparameter Optimization** ve **API testleri** gibi gelişmiş özellikler bulunmaktadır.

---
## 🚀 Kurulum
Aşağıdaki adımları takip ederek projeyi çalıştırabilirsiniz:

### 1️⃣ Gerekli Bağımlılıkları Yükleyin
Aşağıdaki komutu çalıştırarak proje için gereken tüm bağımlılıkları yükleyin:
```sh
pip install -r requirements.txt
```

### 2️⃣ Flask Uygulamasını Başlatın
API'yi çalıştırmak için aşağıdaki komutu kullanın:
```sh
python app.py
```
Başarılı bir şekilde çalıştırıldığında, API **http://127.0.0.1:5000/** adresinde hizmet verecektir.

---
## 📌 API Kullanımı
API, POST ve GET isteklerini desteklemektedir.

### 🔹 1. Tahmin API'si (`POST /predict`)
**İstek Formatı:**
```json
{
    "date": "2025-02-07",
    "conversion_count": 100
}
```
**Yanıt Formatı:**
```json
{
    "model_metrics": {
        "mae": 3.13,
        "mape": 226.04,
        "msle": 9.82,
        "rmse": 3.13
    },
    "predictions": [
        {
            "conversion_count": 65,
            "date": "2025-02-03"
        },
        {
            "conversion_count": 74,
            "date": "2025-02-04"
        },
        {
            "conversion_count": 80,
            "date": "2025-02-05"
        },
        {
            "conversion_count": 80,
            "date": "2025-02-06"
        },
        {
            "conversion_count": 92,
            "date": "2025-02-07"
        },
        {
            "conversion_count": 91,
            "date": "2025-02-08"
        },
        {
            "conversion_count": 70,
            "date": "2025-02-09"
        }
    ]
}
```
Bu uç nokta, verilen tarih için bir tahmin döndürür ve aynı zamanda **MAE, RMSE ve MAPE gibi model doğruluk metriklerini** içerir.

### 🔹 2. Ana Sayfa (`GET /`)
API'nin çalıştığını doğrulamak için aşağıdaki isteği yapabilirsiniz:
```sh
curl http://127.0.0.1:5000/
```
Dönen yanıt:
```sh
"Tahmin API'ye hoş geldiniz!"
```

---
## 🔥 Rate Limiting (API Abuse Önleme)
Bu API, **Flask-Limiter** kullanarak kötüye kullanımı önlemek için hız sınırlaması uygular.
- **Dakikada 10 isteğe kadar izin verilir.**
- Hız sınırını aşan istekler **429 Too Many Requests** hatası alır.

---
## 🎯 Hyperparameter Optimization
Bu projede **Optuna** kullanılarak modelin hiperparametrelerini optimize edebilirsiniz.

Hiperparametre optimizasyonu için aşağıdaki kodu çalıştırabilirsiniz:
```python
import optuna

def objective(trial):
    param = {
        "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.01, 0.5),
        "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10.0)
    }
    return param
```
Optuna, en iyi hiperparametre kombinasyonlarını bulmanıza yardımcı olur.

---
## 🧪 Testler
Projenin düzgün çalıştığını doğrulamak için **pytest** kullanarak testleri çalıştırabilirsiniz:
```sh
pytest test_api.py
```
Tüm testler başarılıysa, API'nin hatasız çalıştığını doğrulamış olursunuz.

---
## 📜 Lisans
Bu proje **MIT Lisansı** ile lisanslanmıştır.

