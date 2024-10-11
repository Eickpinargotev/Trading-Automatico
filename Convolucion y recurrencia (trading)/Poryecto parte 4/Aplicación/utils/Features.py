import numpy as np
import pandas as pd

def agregar_atr_rsi(df, periodo_atr, periodo_rsi):
    """
    Agrega columnas ATR y RSI al DataFrame, calculadas por día del año.
    """
    df['prev_close'] = df.groupby('DayOfYear')['close'].shift(1)
    df['TR'] = df.apply(
        lambda row: max(
            row['high'] - row['low'],
            abs(row['high'] - row['prev_close']) if not pd.isna(row['prev_close']) else row['high'] - row['low'],
            abs(row['low'] - row['prev_close']) if not pd.isna(row['prev_close']) else row['high'] - row['low']
        ),
        axis=1
    )
    atr_column = f'ATR{periodo_atr}'
    df[atr_column] = df.groupby('DayOfYear')['TR'].transform(
        lambda x: x.rolling(window=periodo_atr, min_periods=periodo_atr).mean()
    )
    df[atr_column] = df[atr_column].fillna(0).round(4)

    def calculate_rsi(close_series):
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1 / periodo_rsi, min_periods=periodo_rsi, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / periodo_rsi, min_periods=periodo_rsi, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0)

    rsi_column = f'RSI{periodo_rsi}'
    df[rsi_column] = df.groupby('DayOfYear', group_keys=False)['close'].apply(calculate_rsi)
    df[rsi_column] = df[rsi_column].round(2)
    df = df.drop(columns=['prev_close', 'TR'])
    return df

def agregar_max_min(df, n):
    """
    Agrega las columnas 'maximo' y 'minimo' al DataFrame.
    """
    rolling_max = df['high'].rolling(window=2 * n + 1, center=True).max()
    df['maximo'] = (df['high'] == rolling_max).astype(int).fillna(0).astype(int)
    rolling_min = df['low'].rolling(window=2 * n + 1, center=True).min()
    df['minimo'] = (df['low'] == rolling_min).astype(int).fillna(0).astype(int)
    return df

def agregar_columnas_pips_y_direccion(df):
    """
    Agrega dos columnas al DataFrame: 'PipsBlock' y 'BlockDir'.
    - 'PipsBlock' contiene el valor absoluto de la diferencia entre 'open' y 'close' (movimiento en pips).
    - 'BlockDir' indica la dirección del bloque: 1 para alcista (subida), 0 para bajista (bajada).

    Parámetros:
    - df (pd.DataFrame): El DataFrame original que contiene las columnas 'open', 'high', 'close' y 'low'.

    Retorna:
    - pd.DataFrame: El DataFrame con las nuevas columnas agregadas.
    """
    # Calcular la cantidad de pips en el bloque
    df['PipsBlock'] = (df['close'] - df['open']).abs()

    # Determinar la dirección del bloque
    df['BlockDir'] = (df['close'] > df['open']).astype(int)

    return df