import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils.DataLoader import DataProcessor

# Importamos las funciones necesarias para graficar candlesticks manualmente
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.patches import Rectangle

###############################################################################
# Clase TradingStation (usa precios de cierre)
###############################################################################

import numpy as np
import pandas as pd

class TradingStation:
    def __init__(self, df):
        """
        Parámetros:
          - df: DataFrame con velas de 1 minuto. Se espera que contenga las columnas
                ['open', 'high', 'low', 'close'] y variables adicionales (por ejemplo, 
                'media_fast', 'media_slow', 'm_media_fast' y 'm_media_slow').
        """
        self.df = df
        self.n = len(df)
        self.opens  = df['open'].to_numpy()
        self.highs  = df['high'].to_numpy()
        self.lows   = df['low'].to_numpy()
        self.closes = df['close'].to_numpy()
        self.timestamps = df.index.to_numpy()

        self.macro_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close']]
        if 'media_fast' not in self.macro_columns or 'media_slow' not in self.macro_columns:
            raise ValueError("El DataFrame debe contener las columnas 'media_fast' y 'media_slow'.")

        # Rango horario permitido (por ejemplo, de 11 a 13 horas)
        self.allowed_start = 11
        self.allowed_end   = 14

        # Cada paso se basa en la vela completa
        self.current_index = 0
        
        # Avanza hasta la vela en horario permitido y calcula el estado actual
        self.advance_to_allowed_candle()
        if self.current_index < self.n:
            self.current_state = self.get_micro_state()
        else:
            self.current_state = None

    def in_allowed_hours(self, idx):
        """Retorna True si la vela en el índice idx está en el rango permitido."""
        ts = pd.Timestamp(self.timestamps[idx])
        return self.allowed_start <= ts.hour < self.allowed_end

    def advance_to_allowed_candle(self):
        """Avanza current_index hasta encontrar una vela en horario permitido."""
        while self.current_index < self.n and not self.in_allowed_hours(self.current_index):
            self.current_index += 1

    def reset(self):
        """
        Reinicia el entorno, actualiza y retorna el microestado (basado en el precio de cierre de la vela).
        """
        self.current_index = 0
        self.advance_to_allowed_candle()
        if self.current_index < self.n:
            self.current_state = self.get_micro_state()
        else:
            self.current_state = None
        return self.current_state

    def get_micro_state(self):
        """
        Retorna el microestado actual basado en la vela actual:
            [tendencia, diff_fast]
        donde:
            - tendencia: 0 si media_fast > media_slow, 1 en caso contrario.
            - diff_fast: Diferencia entre el precio de cierre y media_fast.
        """
        row = self.df.iloc[self.current_index]
        media_fast = row['media_fast']
        media_slow = row['media_slow']
        close_price = row['close']
        # Se define la tendencia según la relación entre medias
        tendencia = 1 if media_fast > media_slow else 0
        diff_fast = max(-70, min(70, round(close_price - media_fast)))
        return np.array([tendencia, diff_fast])

    def step(self, accion, stop_loss, take_profit):
        if self.current_index >= self.n:
            raise Exception("No hay más datos disponibles.")

        if not self.in_allowed_hours(self.current_index):
            self.advance_to_allowed_candle()
            if self.current_index >= self.n:
                self.current_state = None
                return None, 0, True, {'info': 'No hay más datos (fuera de horario).'}
            self.current_state = self.get_micro_state()
            return self.current_state, 0, False, {'info': 'Fuera de horario, se avanza a la siguiente vela.'}

        info = {'macro_index': self.current_index, 'accion': accion}

        if accion == 'hold':
            reward = 0
            self.current_index += 1
            self.advance_to_allowed_candle()
            if self.current_index < self.n:
                self.current_state = self.get_micro_state()
                done = False
            else:
                self.current_state = None
                done = True
            info['next_macro_index'] = self.current_index
            return self.current_state, reward, done, info

        # Para 'buy' o 'sell'
        entry_macro_idx = self.current_index
        entry_price = self.closes[entry_macro_idx]

        if accion == 'buy':
            sl_level = entry_price - stop_loss
            tp_level = entry_price + take_profit
        elif accion == 'sell':
            sl_level = entry_price + stop_loss
            tp_level = entry_price - take_profit
        else:
            raise ValueError("Acción inválida. Use 'hold', 'buy' o 'sell'.")

        # --- Vectorización de la simulación usando highs y lows para detectar TP y SL ---
        # Se consideran las velas posteriores a la entrada.
        subsequent_highs = self.highs[entry_macro_idx + 1:]
        subsequent_lows  = self.lows[entry_macro_idx + 1:]
        subsequent_closes = self.closes[entry_macro_idx + 1:]  # Útil para el cierre final si no se cumple TP/SL

        if accion == 'buy':
            # En compra, se pierde si el mínimo baja por debajo del nivel SL,
            # y se gana si el máximo supera el nivel TP.
            stop_loss_mask = subsequent_lows <= sl_level
            take_profit_mask = subsequent_highs >= tp_level
        else:  # accion == 'sell'
            # En venta, se pierde si el máximo supera el nivel SL,
            # y se gana si el mínimo baja por debajo del nivel TP.
            stop_loss_mask = subsequent_highs >= sl_level
            take_profit_mask = subsequent_lows <= tp_level

        # Se obtienen los primeros índices en que se cumplen las condiciones.
        idx_stop = np.nonzero(stop_loss_mask)[0]
        idx_tp = np.nonzero(take_profit_mask)[0]

        candidate_indices = []
        if idx_stop.size > 0:
            candidate_indices.append(idx_stop[0])
        if idx_tp.size > 0:
            candidate_indices.append(idx_tp[0])

        if candidate_indices:
            first_idx = min(candidate_indices)
            # El índice real en el DataFrame es:
            resolution_idx = entry_macro_idx + 1 + first_idx
            # Si ambos niveles se alcanzan en la misma vela, se prioriza el stop loss.
            if stop_loss_mask[first_idx]:
                outcome = 'loss'
                exit_price = sl_level
            elif take_profit_mask[first_idx]:
                outcome = 'win'
                exit_price = tp_level
        else:
            # Si ninguna vela activa TP o SL, se cierra la operación en la última vela disponible.
            resolution_idx = self.n - 1
            exit_price = self.closes[-1]
            if (accion == 'buy' and exit_price > entry_price) or (accion == 'sell' and exit_price < entry_price):
                outcome = 'win'
            else:
                outcome = 'loss'

        # Asignación de recompensa
        reward = 2 if outcome == 'win' else -1

        info.update({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'outcome': outcome,
            'entry_macro_index': entry_macro_idx,
            'resolution_idx': resolution_idx
        })

        # Actualizamos el índice actual y el estado.
        self.current_index = resolution_idx + 1
        self.advance_to_allowed_candle()
        if self.current_index < self.n:
            self.current_state = self.get_micro_state()
            done = False
        else:
            self.current_state = None
            done = True

        return self.current_state, reward, done, info


###############################################################################
# Clase TradingEnvironment (incluye visualización sin recursión)
###############################################################################

class TradingEnvironment:
    def __init__(self, csv_path, render=True):
        """
        Parámetros:
          - csv_path: Nombre (o ruta relativa) del CSV (se asume que está en ../Datos)
          - render: Si True, se muestra la gráfica interactiva.
        """
        self.render = render

        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, os.pardir, os.pardir, os.pardir, 'Datos', csv_path)
        timezone = 'Europe/Helsinki'
        data = DataProcessor(data_path, timezone=timezone)
        data.data_processing()
        data.applyInfo()
        self.df = data.data
        self.n = len(self.df)

        if not np.issubdtype(self.df.index.dtype, np.datetime64):
            self.df.index = pd.to_datetime(self.df.index)

        self.trading_station = TradingStation(self.df)
        # Lista para almacenar todas las operaciones del día actual
        self.daily_trades = []
        # Variable para llevar registro del día actual (para reiniciar las operaciones al cambiar de día)
        self.current_day = None

        # Variables para almacenar los elementos gráficos estáticos y dinámicos
        self.static_plotted = False
        self.vertical_line = None
        self.state_text = None
        self.trade_patches = []

        # Creamos la figura y el eje una única vez
        if self.render:
            plt.style.use('dark_background')  # Usa estilo con fondo oscuro y letras claras
            self.fig, self.ax = plt.subplots(figsize=(18, 6))
            plt.ion()
            self.update_chart(self.trading_station.current_index)
            plt.show()

    def update_static_elements(self, subset, day_str):
        """
        Dibuja los elementos estáticos (candlesticks, medias y líneas) y configura el eje.
        Este método se invoca solo una vez por día (o cuando cambia el día).
        """
        # Limpiamos el eje para la nueva jornada
        self.ax.clear()

        # Convertimos las fechas para usarlas en el candlestick
        subset['Date'] = subset.index.map(mdates.date2num)
        ohlc = subset[['Date', 'open', 'high', 'low', 'close']].values

        # Dibujamos las velas (candlestick)
        candlestick_ohlc(self.ax, ohlc, width=0.0005, colorup='green', colordown='red', alpha=0.8)

        # Dibujamos las medias y líneas adicionales
        self.ax.plot(subset['Date'], subset['media_fast'], label='media_fast', color='blue')
        self.ax.plot(subset['Date'], subset['media_slow'], label='media_slow', color='orange')
        self.ax.plot(subset['Date'], subset['media_fast'] + 30, label='media_fast+30', color='cyan', linestyle='--')
        self.ax.plot(subset['Date'], subset['media_fast'] - 30, label='media_fast-30', color='magenta', linestyle='--')

        # Configuramos el eje de fechas
        self.ax.xaxis_date()
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.fig.autofmt_xdate()

        # Título con la fecha actual
        self.ax.set_title(f'Candlestick Chart - Fecha: {day_str}', color='white')

    def update_dynamic_elements(self, current_index):
        """
        Actualiza o agrega los elementos dinámicos:
          - Línea vertical que marca el estado actual.
          - Texto con el estado actual.
          - Los rectángulos de TP y SL de todas las operaciones del día.
        """
        # Eliminar elementos dinámicos previos (si existen)
        if self.vertical_line is not None:
            try:
                self.vertical_line.remove()
            except Exception:
                pass
        if self.state_text is not None:
            try:
                self.state_text.remove()
            except Exception:
                pass
        if self.trade_patches:
            for patch in self.trade_patches:
                patch.remove()
            self.trade_patches = []

        # Agregar la línea vertical en la posición actual
        current_date = mdates.date2num(self.df.index[current_index])
        self.vertical_line = self.ax.axvline(x=current_date, color='yellow', linestyle='--', linewidth=2, label='Estado Actual')

        # Agregar el texto con el estado actual en la esquina superior izquierda
        estado_actual_text = f"Estado Actual: {', '.join(map(str, self.trading_station.current_state))}"
        self.state_text = self.ax.text(0.01, 0.99, estado_actual_text, transform=self.ax.transAxes,
                                       fontsize=12, color='white', verticalalignment='top',
                                       bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

        # Agregar los rectángulos correspondientes a las operaciones acumuladas del día
        for trade in self.daily_trades:
            accion = trade['accion']
            entry_idx = trade['entry_idx']
            res_idx = trade['resolution_idx']
            entry_price = trade['entry_price']
            x_entry = mdates.date2num(self.df.index[entry_idx])
            x_res = mdates.date2num(self.df.index[res_idx])
            width_rect = x_res - x_entry

            if accion == 'buy':
                # Para compra: TP por encima y SL por debajo del precio de entrada
                tp_bottom = entry_price
                tp_height = trade['take_profit']
                sl_bottom = entry_price - trade['stop_loss']
                sl_height = trade['stop_loss']
            elif accion == 'sell':
                # Para venta: TP por debajo y SL por encima del precio de entrada
                tp_bottom = entry_price - trade['take_profit']
                tp_height = trade['take_profit']
                sl_bottom = entry_price
                sl_height = trade['stop_loss']
            else:
                continue

            # Dibujar el rectángulo del take profit (verde con opacidad)
            tp_rect = Rectangle((x_entry, tp_bottom), width_rect, tp_height,
                                facecolor='green', edgecolor='none', alpha=0.3)
            self.ax.add_patch(tp_rect)

            # Dibujar el rectángulo del stop loss (rojo con opacidad)
            sl_rect = Rectangle((x_entry, sl_bottom), width_rect, sl_height,
                                facecolor='red', edgecolor='none', alpha=0.3)
            self.ax.add_patch(sl_rect)

            # Guardamos los parches para poder eliminarlos en la próxima actualización
            self.trade_patches.append(tp_rect)
            self.trade_patches.append(sl_rect)

    def update_chart(self, current_index):
        """
        Actualiza la gráfica mostrando las velas y líneas del día (entre 10:30 y 16:00)
        y agrega/actualiza los elementos dinámicos:
          - La línea vertical que indica el estado actual.
          - El texto del estado.
          - Las operaciones acumuladas del día.
        Si se cambia de día, se redibuja toda la gráfica.
        """
        if not self.render:
            return

        # Determinamos el día de la vela actual
        new_day = self.df.index[current_index].date()
        # Si cambia de día o no se han dibujado los elementos estáticos aún,
        # filtramos los datos para el día actual y el rango de horas 10:30 a 16:00.
        if (self.current_day != new_day) or (not self.static_plotted):
            self.current_day = new_day
            # Reiniciamos la lista de operaciones al cambiar de día
            self.daily_trades = []
            self.static_plotted = False

            # Filtrar el DataFrame para el día actual
            subset = self.df[self.df.index.date == new_day]
            # Luego, filtramos para el rango horario de 10:30 a 16:00
            subset = subset.between_time("10:30", "16:00").copy()
            if subset.empty:
                return

            day_str = subset.index[0].strftime("%Y-%m-%d")
            # Dibujamos los elementos estáticos
            self.update_static_elements(subset, day_str)
            self.static_plotted = True

        # Actualizamos los elementos dinámicos (línea, texto y operaciones)
        self.update_dynamic_elements(current_index)
        self.fig.canvas.draw_idle()

    def reset(self):
        """Reinicia el entorno, actualiza la gráfica y retorna el microestado."""
        state = self.trading_station.reset()
        # Reiniciamos las operaciones del día
        self.daily_trades = []
        # Forzamos que se redibuje la parte estática en caso de cambio de jornada
        self.static_plotted = False
        if self.render:
            self.update_chart(self.trading_station.current_index)
        return state

    def step(self, accion, stop_loss, take_profit):
        """
        Ejecuta la acción, actualiza la gráfica y retorna
        (nuevo_estado, recompensa, done, info).
        Además, si la acción es 'buy' o 'sell', almacena la operación para graficarla.
        """
        state, reward, done, info = self.trading_station.step(accion, stop_loss, take_profit)
        if accion in ['buy', 'sell']:
            if len(info.keys()) > 1:
                trade = {
                    'accion': accion,
                    'entry_idx': info['entry_macro_index'],
                    'resolution_idx': info['resolution_idx'],
                    'entry_price': info['entry_price'],
                    'exit_price': info['exit_price'],
                    'take_profit': take_profit,
                    'stop_loss': stop_loss
                }
                # Se agrega la operación a la lista de operaciones del día
                self.daily_trades.append(trade)
            else:
                print(info)
        # Para la acción 'hold' u otra, no se modifica la lista de operaciones

        if self.render and not done:
            self.update_chart(self.trading_station.current_index)
        return state, reward, done, info
