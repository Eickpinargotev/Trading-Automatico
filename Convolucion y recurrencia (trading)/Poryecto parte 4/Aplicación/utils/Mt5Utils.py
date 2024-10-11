# Archivo de funciones: utils.py
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from utils.Features import agregar_max_min, agregar_atr_rsi, agregar_columnas_pips_y_direccion


def get_last_candles(symbol, timeframe, num_candles, columns_order_=None):
    """Obtiene las últimas 'num_candles' velas para el símbolo y timeframe especificados."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles + 14)  # 13 velas extras para el ATR y RSI
    if rates is None:
        print(f"No se pudieron obtener las velas para {symbol}.")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.index = df['time']
    df.index = pd.to_datetime(df.index)  # Convertir a datetime
    df = df.drop('time', axis=1)
    df.index = df.index.tz_localize('Etc/GMT-3')
    df.index = df.index.tz_convert('Etc/GMT+5').tz_localize(None)
    df['DayOfYear'] = df.index.dayofyear
    df['Hour'] = df.index.hour
    df = agregar_max_min(df, 4)  # agregar máximos y mínimos
    df = agregar_atr_rsi(df, 14, 14)  # agregar ATR y RSI
    df= agregar_columnas_pips_y_direccion(df)
    df = df[columns_order_]
    return df[13:-1]  # Eliminar las 13 primeras velas para el cálculo del ATR y RSI

def open_order(symbol, order_type, stop_loss_pips, take_profit_pips, risk_percentage, balance):
    """Abre una orden de compra o venta con riesgo calculado automáticamente."""
    point = mt5.symbol_info(symbol).point
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    deviation = 20
    
    # Calcular el stop loss y take profit
    if order_type == mt5.ORDER_TYPE_BUY:
        sl = price - stop_loss_pips
        tp = price + take_profit_pips
    else:
        sl = price + stop_loss_pips
        tp = price - take_profit_pips

    # Calcular el tamaño del lote basado en el riesgo
    symbol_info = mt5.symbol_info(symbol)
    risk_ticks = abs(sl - price) / symbol_info.trade_tick_size
    lot_size = ((risk_percentage / 100) * balance) / (risk_ticks * symbol_info.trade_tick_value)
    lot_size = round(lot_size, 2)
    
    # Crear la solicitud de apertura de orden
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": 123456,
        "comment": "Python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Enviar la orden
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Falló al abrir la orden: {result.retcode}")
        #https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes
    else:
        print(f"Orden abierta: {result.order}")
