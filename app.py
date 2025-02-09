from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, mean_squared_log_error
from pydantic import BaseModel, ValidationError
import logging
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Any, List

# Flask uygulamasını başlat
app = Flask(__name__)

# Modeli yükle
try:
    model = joblib.load("prophet_model.pkl")
except FileNotFoundError:
    model = None

# Eğitim verisini yükle
try:
    df_train = pd.read_csv("ai_task_data.csv")
    df_train["date"] = pd.to_datetime(df_train["date"])
except Exception as e:
    df_train = None
    logging.error(f"Error loading training data: {e}")

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic modelini oluştur
class PredictionRequest(BaseModel):
    date: str

# Veri temizliği: outliers temizleme
def clean_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    z_scores = stats.zscore(df[column])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3)  # 3'ten büyük Z-skorlarını temizle
    return df[filtered_entries]

df_train = clean_outliers(df_train, "conversion_count") if df_train is not None else None

# Özel hata sınıfları
class ModelNotFoundError(Exception):
    pass

class DataLoadingError(Exception):
    pass

class InvalidInputError(Exception):
    pass

@app.route("/predict", methods=["POST"])
def predict() -> Any:
    try:
        data = request.get_json()
        
        if not data:
            raise InvalidInputError("Request body is empty or not in proper format")
        
        try:
            request_data = PredictionRequest(**data)
        except ValidationError as e:
            logger.error(f"Validation Error: {e}")
            raise InvalidInputError(f"Geçersiz veri: {e}")

        if model is None:
            raise ModelNotFoundError("Model not found or failed to load")

        if df_train is None:
            raise DataLoadingError("Training data could not be loaded properly")

        date_str = request_data.date
        logger.info(f"Gelen veri: {data}")
        date = pd.to_datetime(date_str)

        # Gelecek 7 gün için tahmin yapılacak dataframe oluştur
        future = model.make_future_dataframe(periods=7)
        future["day_of_week"] = future["ds"].dt.dayofweek
        future["month"] = future["ds"].dt.month

        # Lag ve hareketli ortalama değişkenlerini ekle
        for lag in [3, 7, 15]:
            future[f"lag_{lag}"] = np.nan

        future["rolling_mean_7"] = np.nan
        future["rolling_mean_15"] = np.nan

        # Geçmiş verilerden lag değerlerini ve hareketli ortalamaları hesapla
        for i in range(len(future)):
            if i >= 3:
                future.loc[i, "lag_3"] = df_train["conversion_count"].iloc[i - 3]
            if i >= 7:
                future.loc[i, "lag_7"] = df_train["conversion_count"].iloc[i - 7]
            if i >= 15:
                future.loc[i, "lag_15"] = df_train["conversion_count"].iloc[i - 15]

            if i >= 7:
                future.loc[i, "rolling_mean_7"] = np.mean(df_train["conversion_count"].iloc[i - 7 : i])
            if i >= 15:
                future.loc[i, "rolling_mean_15"] = np.mean(df_train["conversion_count"].iloc[i - 15 : i])

        future.fillna(df_train["conversion_count"].mean(), inplace=True)

        # Model ile tahmin yap
        forecast = model.predict(future)

        # 7 günlük tahmin verisini al
        forecast_7_days = forecast.tail(7)  # Son 7 günü alıyoruz

        # Model metriklerini hesapla
        y_true = df_train["conversion_count"].iloc[-len(forecast_7_days):].values
        y_pred = forecast_7_days["yhat"].values

        # Logaritma dönüşümü (log1p: log(1 + x) dönüşümü)
        y_true_log = np.log1p(np.maximum(y_true, 0))  # Gerçek değerlerin log dönüşümü
        y_pred_log = np.log1p(np.maximum(y_pred, 0))  # Tahmin edilen değerlerin log dönüşümü

        # MAPE'yi hesapla (log dönüşümü sonrası)
        non_zero_indices = y_true_log != 0  # Gerçek değerlerin sıfır olmadığı indeksleri bul
        y_true_log_non_zero = y_true_log[non_zero_indices]
        y_pred_log_non_zero = y_pred_log[non_zero_indices]

        if len(y_true_log_non_zero) > 0:
            mape = np.mean(np.abs((y_true_log_non_zero - y_pred_log_non_zero) / y_true_log_non_zero)) * 100
        else:
            mape = None  # Eğer geçerli veri yoksa, MAPE hesaplanamaz

        # RMSE'yi hesapla (log dönüşümü sonrası)
        rmse = np.sqrt(np.mean((y_true_log_non_zero - y_pred_log_non_zero) ** 2))

        # MAE'yi hesapla (log dönüşümü sonrası)
        mae = np.mean(np.abs(y_true_log_non_zero - y_pred_log_non_zero))

        # MSLE'yi hesapla (log dönüşümü sonrası)
        msle = np.mean((y_true_log_non_zero - y_pred_log_non_zero) ** 2)

        model_metrics = {
            "mape": round(mape, 2) if mape is not None else None,
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "msle": round(msle, 2)
        }

        # Tahmin sonuçlarını istenen formata çevir
        predictions = [
            {"date": str(row["ds"].date()), "conversion_count": round(row["yhat"])}
            for _, row in forecast_7_days.iterrows()
        ]

        logger.info(f"Prediction Result: {predictions}")
        logger.info(f"Model Metrics: {model_metrics}")

        return jsonify({"predictions": predictions, "model_metrics": model_metrics})

    except ModelNotFoundError as e:
        logger.error(f"Model Error: {e}")
        return jsonify({"error": "Model is not available. Please check the server configuration."}), 500
    except DataLoadingError as e:
        logger.error(f"Data Loading Error: {e}")
        return jsonify({"error": "Training data could not be loaded. Please check the dataset."}), 500
    except InvalidInputError as e:
        logger.error(f"Invalid Input: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500

@app.route("/")  
def home() -> str:  
    return "Tahmin API'ye hoş geldiniz!"  

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
