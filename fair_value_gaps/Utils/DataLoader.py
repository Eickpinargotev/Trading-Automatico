import pandas as pd

class DataProcessor:
    def __init__(self, data_path, news_path=None,timezone='Europe/Helsinki',timezone_destino='America/New_York'):
        # Inicializamos los atributos de datos y noticias
        self.data_path = data_path
        self.news_path = news_path
        self.data = None
        self.news = None
        self.timezone = timezone
        self.timezone_destino = timezone_destino

    # Método para cargar datos
    def data_loader(self):
        self.data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        self.data = self.data[['open', 'high', 'low', 'close']]#, 'tick_volume']]
        if self.data.index.duplicated().any():
            print('Duplicated index found. Removing duplicates...')
            self.data = self.data[~self.data.index.duplicated(keep='last')]
        # Cambiar zona horaria
        self.data.index = self.data.index.tz_localize(self.timezone).tz_convert(self.timezone_destino).tz_localize(None)

    # Métodos de procesamiento de datos
    def descarte_dias(self):
        self.data['doy'] = self.data.index.dayofyear
        self.data = self.data[~self.data['doy'].isin([self.data.index.dayofyear[0], self.data.index.dayofyear[-1]])]
        self.data.drop(columns='doy', inplace=True)

    def take_hours(self, df, zone=list(range(8, 12))):
        df['Hour'] = df.index.hour
        df = df[df['Hour'].isin(zone)]
        return df.drop(columns='Hour')

    def japanese_candlesticks_features(self):
        self.data['body'] = self.data['close'] - self.data['open']

    # Método principal para procesamiento de datos
    def data_processing(self,hours=list(range(8, 12)),body=False):
        self.data_loader()
        self.descarte_dias()#Eliminamos el primer y ultimo día
        self.data = self.take_hours(self.data,hours)#Tomamos el rango de tiempo desde las 8 hasta las 12
        if body:
            self.japanese_candlesticks_features()
