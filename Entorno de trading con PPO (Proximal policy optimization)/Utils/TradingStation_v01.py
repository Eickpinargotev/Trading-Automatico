import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

from Utils.DataLoader import DataProcessor

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.patches import Rectangle

class TradingEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,csv_path, render_mode=None,sl=30,tp=40,balance=10000,risk=200):
        self.observation_space = spaces.Dict({
        "distance": spaces.Box(low=-70.0, high=70.0, shape=(1,), dtype=np.float32),
        "balance": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
        "trend": spaces.Discrete(2),
        "daily_balance": spaces.Box(low=-500.0, high=500.0, shape=(1,), dtype=np.float32),
        "fase": spaces.Box(low=1, high=2, shape=(1,), dtype=np.int32)  # Nueva variable "fase"
        })
        # We have 3 actions, corresponding to "buy", "sell", "hold"
        self.action_space = spaces.Discrete(3)
        self._action_to_orden = {
            0: "buy",
            1: "sell",
            2: "hold",
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        # -------------------------------------------------------------------------------------------------------
        """
        Parámetros:
          - df: DataFrame con velas de 1 minuto. Se espera que contenga las columnas
                ['open', 'high', 'low', 'close'] y variables adicionales (por ejemplo, 
                'media_fast', 'media_slow', 'm_media_fast' y 'm_media_slow').
        """
        self._inicializar(csv_path,sl,tp,balance,risk)

    def reset(self, seed=None, options=None):
        # No usamos valores ramdons
        # self.current_index = 0 #No reiniciamos el dataframe, de esto se encarga la condicional de truncamiento en step()
        self._balance = self.balance
        self.current_index = np.random.choice(self.day_indices) # Inicializamos current_index en un día aleatorio
        self._advance_to_allowed_candle() # Avanzamos al primer paso en horario permitido
        self._get_micro_state() # Obtenemos el microestado actual

        # Reiniciamos la pérdida diaria y guardamos el día actual
        self.daily_balance = 0.0
        self.last_trade_day = pd.Timestamp(self.timestamps[self.current_index]).date()
        self.phase = 1  # Inicializamos la fase en 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        self.daily_trades = []  # Reiniciar operaciones diarias

        return observation, info
    
    def step(self, action):
        orden = self._action_to_orden[action] # Compra, Venta o Mantener
        if orden == 'hold':
            self.current_index += 1
            self._advance_to_allowed_candle()
            self._get_micro_state()
            terminated = self.current_index >= self.n - 1
            return self._get_obs(), 0, terminated, False, self._get_info()

        elif orden in ['buy', 'sell']:
            entry_macro_idx = self.current_index
            entry_price = self.closes[entry_macro_idx]
            # --- Vectorización de la simulación usando highs y lows para detectar TP y SL ---
            # Se consideran las velas posteriores a la entrada.
            subsequent_highs = self.highs[entry_macro_idx + 1:]
            subsequent_lows  = self.lows[entry_macro_idx + 1:]
            # Determinar niveles TP y SL y generar las condiciones
            if orden == 'buy':
                sl_level = entry_price - self.stop_loss
                tp_level = entry_price + self.take_profit
                cond_stop = subsequent_lows <= sl_level
                cond_tp = subsequent_highs >= tp_level
            else:  # 'sell'
                sl_level = entry_price + self.stop_loss
                tp_level = entry_price - self.take_profit
                cond_stop = subsequent_highs >= sl_level
                cond_tp = subsequent_lows <= tp_level

            # Obtener el primer índice donde se cumple cada condición o None si no se cumple
            first_stop = np.argmax(cond_stop) if cond_stop.any() else None
            first_tp = np.argmax(cond_tp) if cond_tp.any() else None

            if first_stop is None and first_tp is None:
                # Si ninguna condición se cumple, se cierra en la última vela
                resolution_idx = self.n - 1
                exit_price = self.closes[-1]
                if (orden == 'buy' and exit_price > entry_price) or (orden == 'sell' and exit_price < entry_price):
                    outcome = 'win'
                else:
                    outcome = 'loss'
            else:
                # Se asigna first_idx con el valor finito disponible o el menor entre ambos
                if first_stop is None:
                    first_idx = first_tp
                elif first_tp is None:
                    first_idx = first_stop
                else:
                    first_idx = min(first_stop, first_tp)

                resolution_idx = entry_macro_idx + 1 + first_idx
                # Si se cumplen ambas condiciones en la misma vela, se prioriza el stop loss
                if first_stop is not None and first_idx == first_stop:
                    outcome = 'loss'
                    exit_price = sl_level
                else:
                    outcome = 'win'
                    exit_price = tp_level

            # Actualizar el balance y la pérdida diaria de forma vectorizada
            if outcome == 'win':
                self._balance += self.risk * self.ratio_rb
                self.daily_balance += self.risk * self.ratio_rb
            else:
                self._balance -= self.risk
                self.daily_balance -= self.risk
            
            # Antes de la resolución, guardar el día de la operación:
            trade_day = pd.Timestamp(self.timestamps[entry_macro_idx]).date()

            # Asignación de nueva observación, recompensa y estado de terminación y truncamiento.
            self.current_index = resolution_idx +1 if resolution_idx < self.n-1 else resolution_idx # Establecemos el índice actual para calcular _get_obs()
            self._advance_to_allowed_candle()
            self._manage_daily_balance()
            self.current_index = 0 if self.current_index >= self.n - 1 else self.current_index#Indica el final del Dataframe y se reinicia.
            self._get_micro_state()

            # Obtener el día actual luego de la resolución:
            current_day = pd.Timestamp(self.timestamps[self.current_index]).date()
            
            # Si el balance supera los self.balance * 1.1, se termina el episodio, si se termina el DataFrame, se reinicia el dataFrame el episodio.
            done = (
            (self._balance >= self.balance * 1.05 and self.phase==2 ) or
            self._balance <= self.balance * 0.9 or 
            self.daily_balance <= -self.balance*0.05
            )
            
            # Asignar la recompensa según el resultado
            if done: 
                reward_value = 15 if self._balance >= self.balance * 1.05 else -10
            elif self.phase==1 and self._balance >= self.balance * 1.1:
                reward_value = 10
                self._advance_day(current_day,trade_day)  # Cambia al siguiente día, reinicia el balance y actualiza la fase
            else:
                reward_value = self.ratio_rb if outcome == 'win' else -1
            
            # Aquí agregamos la operación a la lista daily_trades
            trade_info = {
                'accion': orden,
                'entry_idx': entry_macro_idx,
                'resolution_idx': resolution_idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'take_profit': self.take_profit,
                'stop_loss': self.stop_loss
            }
            self.daily_trades.append(trade_info)

            return self._get_obs(),reward_value, done, False, self._get_info()
        else:
            raise ValueError("Acción inválida. Use 'hold', 'buy' o 'sell'.")

    def _get_obs(self):
        return {
        "distance": np.array([self._agent_diff], dtype=np.float32),
        "balance": np.array([self._balance], dtype=np.float32),
        "trend": self._agent_trend,
        "daily_balance": np.array([self.daily_balance], dtype=np.float32),
        "fase": np.array([self.phase], dtype=np.int32)  # Nueva variable fase
        }
    
    def _get_info(self):
        return {"phase":self.phase,"balance":self._balance,"distance": self._agent_diff, "trend": self._agent_trend, "current_index": self.current_index}
    
    def _in_allowed_hours(self, idx):
        """Retorna True si la vela en el índice idx está en el rango permitido."""
        ts = pd.Timestamp(self.timestamps[idx])
        return self.allowed_start <= ts.hour < self.allowed_end

    def _advance_to_allowed_candle(self):
        """Avanza current_index hasta la siguiente vela en horario permitido usando búsqueda binaria."""
        if self.current_index >= self.n - 1:
            return
        pos = np.searchsorted(self.allowed_indices, self.current_index)
        if pos < len(self.allowed_indices):
            self.current_index = self.allowed_indices[pos]
    
    def _get_micro_state(self):
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
        self._agent_trend = 1 if media_fast > media_slow else 0
        self._agent_diff = max(-70, min(70, round(close_price - media_fast,2)))

    def _inicializar(self, csv_path,sl,tp,balance,risk):

        self.df = self._cargar_df(csv_path)
        self.n = len(self.df)
        self.opens  = self.df['open'].to_numpy()
        self.highs  = self.df['high'].to_numpy()
        self.lows   = self.df['low'].to_numpy()
        self.closes = self.df['close'].to_numpy()
        self.timestamps = self.df.index.to_numpy()
        self.macro_columns = [col for col in self.df.columns if col not in ['open', 'high', 'low', 'close']]
        self.required_columns = ['media_slow', 'media_fast'] # Columnas requeridas para el entorno
        if not all(col in self.macro_columns for col in self.required_columns):
            raise ValueError("El DataFrame debe contener las columnas 'media_fast' y 'media_slow'.")

        # Rango horario permitido
        self.allowed_start = 11
        self.allowed_end   = 14

        # valor de stop_loss y take_profit
        self.stop_loss = sl
        self.take_profit = tp
        self.ratio_rb = self.take_profit/self.stop_loss

        # balance de la cuenta
        self.balance = balance
        self.risk = risk

        # Inicialización de variables para el seguimiento de la pérdida diaria
        self.daily_balance = 0.0
        self.last_trade_day = None

        # Crear una lista con los índices correspondientes al inicio de cada día
        day_indices_step1 = np.where(~self.df.index.normalize().duplicated())[0]
        self.day_indices = day_indices_step1[:-20] if len(day_indices_step1) > 20 else day_indices_step1# Todos los días menos los ultimos 20
        self.current_index = np.random.choice(self.day_indices) # Inicializamos current_index en un día aleatorio

        # Precomputar los índices en los que la hora está en el rango permitido
        allowed_hours = pd.to_datetime(self.timestamps).hour
        allowed_mask = (allowed_hours >= self.allowed_start) & (allowed_hours < self.allowed_end)
        self.allowed_indices = np.where(allowed_mask)[0]

    def _cargar_df(self, csv_path):
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, os.pardir, os.pardir, os.pardir, 'Datos', csv_path)
        timezone = 'Europe/Helsinki'
        data = DataProcessor(data_path, timezone=timezone)
        data.data_processing()
        data.applyInfo()
        return data.data
    
    def _manage_daily_balance(self):
        """
        Actualiza la pérdida diaria y reinicia el balance diario si es un nuevo día.
        """
        current_day = pd.Timestamp(self.timestamps[self.current_index]).date()
        if self.last_trade_day is None or current_day != self.last_trade_day:
            self.daily_balance = 0.0
            self.last_trade_day = current_day

    def render(self, mode=None):
        """
        Renderiza el entorno usando matplotlib.
        Compatible con Stable Baselines3.
        
        Parámetros:
        - mode: "human" para mostrar interactivamente o "rgb_array" para retornar la imagen como np.array.
                Si no se especifica, se utiliza self.render_mode.
        """
        if mode is None:
            mode = self.render_mode
        if mode is None:
            return

        # Asegurarse de que el índice del DataFrame sea datetime
        if not np.issubdtype(self.df.index.dtype, np.datetime64):
            self.df.index = pd.to_datetime(self.df.index)

        # Inicializar la figura y los ejes si aún no existen
        if not hasattr(self, "fig") or self.fig is None:
            plt.style.use("dark_background")
            self.fig, self.ax = plt.subplots(figsize=(18, 6))
            plt.ion()
            self.current_day = None
            self.static_plotted = False
            self.vertical_line = None
            self.state_text = None
            self.trade_patches = []
            self.daily_trades = []

        current_index = self.current_index
        current_time = self.df.index[current_index]
        new_day = current_time.date()

        # Actualizar elementos estáticos si es un nuevo día o si aún no se han ploteado
        if (self.current_day != new_day) or (not self.static_plotted):
            self.current_day = new_day
            self.daily_trades = []  # Reiniciar operaciones diarias
            self.static_plotted = False

            # Filtrar el DataFrame para el día actual y para el rango de horas deseado
            subset = self.df[self.df.index.date == new_day]
            subset = subset.between_time("10:30", "16:00").copy()
            if subset.empty:
                return

            day_str = subset.index[0].strftime("%Y-%m-%d")
            # Dibujar elementos estáticos: candlesticks, medias y líneas de referencia
            self.ax.clear()
            subset["Date"] = subset.index.map(mdates.date2num)
            ohlc = subset[["Date", "open", "high", "low", "close"]].values
            candlestick_ohlc(self.ax, ohlc, width=0.0005, colorup="green", colordown="red", alpha=0.8)
            self.ax.plot(subset["Date"], subset["media_fast"], label="media_fast", color="blue")
            self.ax.plot(subset["Date"], subset["media_slow"], label="media_slow", color="orange")
            self.ax.plot(subset["Date"], subset["media_fast"] + 30, label="media_fast+30", color="cyan", linestyle="--")
            self.ax.plot(subset["Date"], subset["media_fast"] - 30, label="media_fast-30", color="magenta", linestyle="--")
            self.ax.xaxis_date()
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            self.fig.autofmt_xdate()
            self.ax.set_title(f'Candlestick Chart - Fecha: {day_str}', color="white")
            self.ax.legend()
            self.static_plotted = True

        # Actualizar elementos dinámicos: remover línea vertical, texto y parches previos
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

        # Agregar la línea vertical que indica la vela actual
        current_date = mdates.date2num(current_time)
        self.vertical_line = self.ax.axvline(x=current_date, color="yellow", linestyle="--", linewidth=2)

        # Agregar texto con el estado actual obtenido de _get_obs()
        obs = self._get_obs()
        state_text_str = f"phase={self.phase}, daily_balance={obs['daily_balance'][0]}, balance={obs['balance']}, trend={obs['trend']}, diff={obs['distance'][0]}"
        self.state_text = self.ax.text(0.01, 0.99, state_text_str, transform=self.ax.transAxes,
                                    fontsize=12, color="white", verticalalignment="top",
                                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))
        # Dibujar los rectángulos de trades (TP y SL) de las operaciones acumuladas
        for trade in self.daily_trades:
            accion = trade['accion']
            entry_idx = trade['entry_idx']
            res_idx = trade['resolution_idx']
            entry_price = trade['entry_price']
            x_entry = mdates.date2num(self.df.index[entry_idx])
            x_res = mdates.date2num(self.df.index[res_idx])
            width_rect = x_res - x_entry

            if accion == 'buy':
                tp_bottom = entry_price
                tp_height = trade['take_profit']
                sl_bottom = entry_price - trade['stop_loss']
                sl_height = trade['stop_loss']
            elif accion == 'sell':
                tp_bottom = entry_price - trade['take_profit']
                tp_height = trade['take_profit']
                sl_bottom = entry_price
                sl_height = trade['stop_loss']
            else:
                continue

            tp_rect = Rectangle((x_entry, tp_bottom), width_rect, tp_height,
                                facecolor="green", edgecolor="none", alpha=0.3)
            self.ax.add_patch(tp_rect)
            sl_rect = Rectangle((x_entry, sl_bottom), width_rect, sl_height,
                                facecolor="red", edgecolor="none", alpha=0.3)
            self.ax.add_patch(sl_rect)
            self.trade_patches.append(tp_rect)
            self.trade_patches.append(sl_rect)

        self.fig.canvas.draw_idle()


        if mode == "human":
            plt.pause(0.001)
            return None
        elif mode == "rgb_array":
            self.fig.canvas.draw()
            width, height = self.fig.canvas.get_width_height()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape((height, width, 3))
            return image
        else:
            return None

    def _render_frame(self):
        self.render("human")
    
    def _advance_day(self, current_day, trade_day):
        """
        Cambia al siguiente día, reinicia el balance a su valor original y actualiza la fase a 2.
        """
        # Reiniciar el balance y actualizar la fase
        self._balance = self.balance
        self.phase = 2

        if current_day == trade_day:
            # Obtener la fecha actual (día de la vela actual)
            current_day = pd.Timestamp(self.timestamps[self.current_index]).date()

            # Extraer las fechas correspondientes a los índices precomputados (inicio de cada día)
            day_dates = pd.to_datetime(self.timestamps[self.day_indices]).date

            # np.searchsorted devuelve la posición donde se debería insertar current_day para mantener el orden.
            # Usamos side='right' para obtener la primera posición con un día mayor a current_day.
            pos = np.searchsorted(day_dates, current_day, side='right')

            # Si pos es menor que la longitud, encontramos el siguiente día; de lo contrario, reiniciamos al primer día.
            new_index = self.day_indices[pos] if pos < len(day_dates) else self.day_indices[0]

            self.current_index = new_index
            self._advance_to_allowed_candle()
            self._manage_daily_balance()