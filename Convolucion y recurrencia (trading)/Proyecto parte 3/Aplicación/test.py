# Archivo principal: main.py
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
from utils.AccessKeys import login1_erick, password1_erick, server_erick  # Importar claves desde access_keys.py

# Configuración de parámetros principales
SYMBOL = "US30.cash"  # Símbolo de instrumento financiero a operar
TIMEFRAME = mt5.TIMEFRAME_M1  # Marco temporal de 1 minuto
NUM_CANDLES = 300  # Número de velas utilizadas para el análisis
STOP_LOSS_PIPS = 30  # Nivel de stop loss en pips
TAKE_PROFIT_MULTIPLIER = 3  # Multiplicador del take profit en relación al stop loss

# Columnas a normalizar y orden de columnas que espera el modelo
columnas_a_normalizar = ['open', 'high', 'low', 'close', 'tick_volume', 'Hour', 'ATR14', 'RSI14']  # Columnas a escalar
columns_order = ['open', 'high', 'low', 'close', 'tick_volume', 'Hour', 'maximo', 'minimo', 'ATR14', 'RSI14']  # Orden de columnas requerido por el modelo

# Cargar y configurar el escalador MinMaxScaler utilizando datos históricos
try:
    Buys_sample = pd.read_csv('Proyecto parte 3/Buys.csv')  # Cargar datos de ejemplo para ajustar el escalador
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
        data = scaler.transform(df[columnas_a_normalizar])

        # Solicitar al usuario si desea realizar una compra o una venta
        while True:
            user_input = input("Ingrese '1' para realizar una operación de compra o venta, o presione Enter para esperar: ")
            if user_input == '1':
                # Solicitar al usuario si es compra o venta
                operation_type = input("Ingrese 'buy' para compra o 'sell' para venta: ").strip().lower()
                if operation_type == 'buy':
                    print("Predicción de Compra detectada. Abriendo orden de compra.")
                    open_order(SYMBOL, mt5.ORDER_TYPE_BUY, STOP_LOSS_PIPS, STOP_LOSS_PIPS * TAKE_PROFIT_MULTIPLIER, 0.2, 10000)
                    break
                elif operation_type == 'sell':
                    print("Predicción de Venta detectada. Abriendo orden de venta.")
                    open_order(SYMBOL, mt5.ORDER_TYPE_SELL, STOP_LOSS_PIPS, STOP_LOSS_PIPS * TAKE_PROFIT_MULTIPLIER, 0.2, 10000)
                    break
                else:
                    print("Entrada no válida. Por favor, ingrese 'buy' o 'sell'.")
            else:
                print("Esperando entrada válida...")

        # Esperar hasta el final del minuto antes de iniciar el próximo ciclo
        elapsed = time.time() - start_time
        sleep_time = max(60 - elapsed, 0)
        time.sleep(sleep_time)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Ejecución del script principal
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
    finally:
        mt5.shutdown()