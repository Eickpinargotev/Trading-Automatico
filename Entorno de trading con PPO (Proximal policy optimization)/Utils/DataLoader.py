import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, data_path,timezone='Europe/Helsinki'):
        # Inicializamos los atributos de datos y noticias
        self.data_path = data_path
        self.data = None
        self.timezone = timezone

    # Método para cargar datos
    def data_loader(self):
        self.data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        self.data = self.data[['open', 'high', 'low', 'close']]
        if self.data.index.duplicated().any():
            print('Duplicated index found. Removing duplicates...')
            self.data = self.data[~self.data.index.duplicated(keep='last')]
        # Cambiar zona horaria
        if not self.timezone:
            self.data.index = self.data.index.tz_localize(self.timezone).tz_convert('America/Guayaquil').tz_localize(None)
        else:
            self.data.index = self.data.index.tz_localize(self.timezone).tz_convert('America/New_York').tz_localize(None)

    def descarte_dias(self):
        self.data['doy'] = self.data.index.dayofyear
        self.data = self.data[~self.data['doy'].isin([self.data.index.dayofyear[0], self.data.index.dayofyear[-1]])]
        self.data.drop(columns='doy', inplace=True)
    
    def applyInfo(self):
        # Media móvil rápida (periodo = 14) aplicada a la columna 'close'
        self.data['media_fast'] = self.data['close'].rolling(window=14).mean().shift(0)
        # Media móvil lenta (periodo = 100) aplicada a la columna 'close'
        self.data['media_slow'] = self.data['close'].rolling(window=100).mean().shift(0)
        # Pendiente de media_fast
        self.data['m_media_fast'] = (self.data['media_fast'] - self.data['media_fast'].shift(1))/10
        # Pendiente de media_slow
        self.data['m_media_slow'] = (self.data['media_slow'] - self.data['media_slow'].shift(1))/10
        # Eliminamos los valores nulos
        self.data.dropna(inplace=True)


    # Método principal para procesamiento de datos
    def data_processing(self):
        self.data_loader()
        self.descarte_dias()#Eliminamos el primer y ultimo día