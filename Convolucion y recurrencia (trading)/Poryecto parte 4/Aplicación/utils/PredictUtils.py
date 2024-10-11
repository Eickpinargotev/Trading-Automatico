import numpy as np
import pandas as pd

def prepare_data(df, scaler_trained=None, columns_=None):
    try:
        df[columns_] = scaler_trained.transform(df[columns_])
    except Exception as e:
        print(f'Hubo un error al cargar el scaler o cargar las columnas: {e}')
    return np.expand_dims(df.values, axis=0)

def predict_operations(data, Buys_model, Sells_model):
    """Realiza las predicciones de compra y venta."""
    y_pred_prob_Buys = Buys_model.predict(data)[0]
    y_pred_prob_Sells = Sells_model.predict(data)[0]
    buy_pred = (y_pred_prob_Buys >= 0.7).astype(int).reshape(-1)
    sell_pred = (y_pred_prob_Sells >= 0.7).astype(int).reshape(-1)
    return y_pred_prob_Buys, y_pred_prob_Sells, buy_pred, sell_pred