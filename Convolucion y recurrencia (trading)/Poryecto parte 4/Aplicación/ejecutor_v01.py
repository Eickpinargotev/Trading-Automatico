import time 
import signal
import sys
import MetaTrader5 as mt5
import os
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Importar funciones desde utils
from utils.Mt5Utils import get_last_candles, open_order 
from utils.PredictUtils import prepare_data, predict_operations  
from utils.AccessKeys import login1_erick, password1_erick, server_erick  # Importar claves desde AccessKeys.py

# Configuración de parámetros principales
SYMBOL = "US30.cash"  # Símbolo de instrumento financiero a operar
TIMEFRAME = mt5.TIMEFRAME_M1  # Marco temporal de 1 minuto
NUM_CANDLES = 300  # Número de velas utilizadas para el análisis
STOP_LOSS_PIPS = 30  # Nivel de stop loss en pips
TAKE_PROFIT_MULTIPLIER = 3  # Multiplicador del take profit en relación al stop loss

# Columnas a normalizar y orden de columnas que espera el modelo
columnas_a_normalizar = ['open', 'high', 'low', 'close', 'tick_volume', 'Hour', 'ATR14', 'RSI14', 'maximo', 'minimo', 'PipsBlock', 'BlockDir']  # Columnas a escalar
columns_order = ['open', 'high', 'low', 'close', 'tick_volume', 'Hour', 'ATR14', 'RSI14', 'maximo', 'minimo', 'PipsBlock', 'BlockDir']  # Orden de columnas requerido por el modelo

# Carga de modelos preentrenados para predicción de operaciones de compra y venta
try:
    Buys_model = load_model(os.path.join(os.path.dirname(__file__), 'models/BuysModel.keras'))  # Cargar modelo de compras
    Sells_model = load_model(os.path.join(os.path.dirname(__file__), 'models/SellsModel.keras'))  # Cargar modelo de ventas
    os.system('cls')
    print("Modelos cargados correctamente.")
except Exception as e:
    # Finalizar el script si ocurre un error al cargar los modelos
    print(f"Error al cargar los modelos: {e}")
    sys.exit(1)

# Cargar y configurar el escalador MinMaxScaler utilizando datos históricos
try:
    Buys_sample = pd.read_csv('Buys.csv')  # Cargar datos de ejemplo para ajustar el escalador
    scaler = MinMaxScaler()  # Inicializar el escalador
    scaler.fit(Buys_sample[columns_order][columnas_a_normalizar])  # Ajustar el escalador con las columnas especificadas
    print("Escalador cargado correctamente.")
except Exception as e:
    # Finalizar el script si ocurre un error al cargar el escalador
    print(f"Error al cargar el escalador: {e}")
    sys.exit(1)

# Manejo de señal para cierre del script de manera segura
def signal_handler(sig, frame):
    print("Cerrando el script...")
    mt5.shutdown()
    sys.exit(0)

# Asignar manejadores de señal para SIGINT y SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    # Inicializar conexión con MetaTrader 5 usando credenciales proporcionadas
    if not mt5.initialize(login=login1_erick, server=server_erick, password=password1_erick):
        print("initialize() failed, error code =", mt5.last_error())
        mt5.shutdown()
        sys.exit(1)
    else:
        print("MetaTrader5 inicializado correctamente.")

    # Ciclo principal de trading
    print("Iniciando el ciclo de trading...")
    while True:
        current_time = datetime.now()
        # Definir ventana de trading (de 8 AM a 2 PM)
        if 8 <= current_time.hour < 14:
            start_time = time.time()
            print(f"Procesando a las {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Obtener las últimas velas para el símbolo especificado
            df = get_last_candles(SYMBOL, TIMEFRAME, NUM_CANDLES, columns_order)
            if df is None:
                # Si la obtención de velas falla, esperar hasta el próximo ciclo
                print("Esperando el próximo ciclo debido a la falla en la obtención de velas.")
                time.sleep(60 - (time.time() - start_time))
                continue

            # Preparar los datos para ser ingresados al modelo
            data = prepare_data(df, scaler, columnas_a_normalizar)
            print(data.shape)
            # Realizar predicciones con los modelos de compra y venta
            b_r, s_r, buy_pred, sell_pred = predict_operations(data, Buys_model, Sells_model)

            # Analizar resultados de predicción y ejecutar operaciones si corresponde
            if buy_pred == 0 and sell_pred == 0:
                print("Predicción insuficiente - Venta: {} Compra: {}".format(s_r, b_r))
            if buy_pred == 1 and sell_pred == 0:
                print("Predicción de Compra detectada. Abriendo orden de compra.")
                open_order(SYMBOL, mt5.ORDER_TYPE_BUY, STOP_LOSS_PIPS, STOP_LOSS_PIPS * TAKE_PROFIT_MULTIPLIER, 0.1, 10000)
            if sell_pred == 1 and buy_pred == 0:
                print("Predicción de Venta detectada. Abriendo orden de venta.")
                open_order(SYMBOL, mt5.ORDER_TYPE_SELL, STOP_LOSS_PIPS, STOP_LOSS_PIPS * TAKE_PROFIT_MULTIPLIER, 0.1, 10000)

            # Esperar hasta el final del minuto antes de iniciar el próximo ciclo
            elapsed = time.time() - start_time
            sleep_time = max(60 - elapsed, 0)
            time.sleep(sleep_time)
        else:
            # Fuera del horario de trading definido
            print("Fuera de horario")
            time.sleep(60)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Ejecución del script principal
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
    finally:
        mt5.shutdown()
