import pytest
import json
import numpy as np
from flask import Flask
from app import app as flask_app  # Flask uygulamasını içe aktar
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import optuna

# Hız sınırlaması mekanizması tanımlanıyor
limiter = Limiter(get_remote_address, app=flask_app, default_limits=["10 per minute"])

test_client = flask_app.test_client()

# Ana sayfa endpoint'inin düzgün çalıştığını test eden fonksiyon
def test_home():
    response = test_client.get("/")
    assert response.status_code == 200
    assert "Tahmin API'ye hoş geldiniz!" in response.data.decode("utf-8")

# Geçerli bir tarih ile tahmin API'sinin yanıtını test eden fonksiyon
def test_predict_valid_date():
    response = test_client.post("/predict", 
                                data=json.dumps({"date": "2025-02-10"}),
                                content_type="application/json")
    assert response.status_code == 200
    data = response.get_json()
    assert "predictions" in data
    assert "model_metrics" in data
    assert isinstance(data["predictions"], list)

# Geçersiz bir tarih formatı gönderildiğinde API'nin uygun hata mesajı döndürdüğünü test eden fonksiyon
def test_predict_invalid_date():
    response = test_client.post("/predict", 
                                data=json.dumps({"date": "invalid-date"}),
                                content_type="application/json")
    assert response.status_code in [400, 500]
    data = response.get_json()
    assert "error" in data

# Eksik alan gönderildiğinde API'nin doğru hata kodunu döndürdüğünü kontrol eden test fonksiyonu
def test_predict_missing_field():
    response = test_client.post("/predict", 
                                data=json.dumps({}),
                                content_type="application/json")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

# Gelecekteki bir tarih için tahmin API'sinin nasıl yanıt verdiğini kontrol eden fonksiyon
def test_predict_future_date():
    response = test_client.post("/predict", 
                                data=json.dumps({"date": "2030-01-01"}),
                                content_type="application/json")
    assert response.status_code == 200
    data = response.get_json()
    assert "predictions" in data
    assert "model_metrics" in data
    assert isinstance(data["predictions"], list)

# Modelin zaman içindeki kaymasını (drift) tespit etmek için test fonksiyonu
def test_model_drift():
    # Modelin önceki tahminlerinin daha güncel değerler olması gerekebilir
    previous_predictions = np.array([65, 74, 80, 80, 92, 91, 70])

    response = test_client.post("/predict", 
                                data=json.dumps({"date": "2025-02-10"}),
                                content_type="application/json")
    assert response.status_code == 200
    data = response.get_json()
    
    # Gelen veriyi sayısal değerlere dönüştür
    current_predictions = np.array([item["conversion_count"] for item in data["predictions"]])
    
    # Eşik değerini önceki tahminlerin standart sapmasına göre belirle
    drift_threshold = np.std(previous_predictions) * 3  
    assert np.abs(np.mean(current_predictions) - np.mean(previous_predictions)) < drift_threshold

# Hiperparametre optimizasyonu için kullanılan fonksiyon
def objective(trial):
    param = {
        "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.01, 0.5),
        "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10.0),
    }
    return param  # Burada model eğitimi ve değerlendirme metrikleri hesaplanabilir

if __name__ == "__main__":
    pytest.main()
