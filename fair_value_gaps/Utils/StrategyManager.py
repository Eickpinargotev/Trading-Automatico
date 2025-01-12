import pandas as pd
import datetime
from datetime import time, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class ApplyStrategy:
    def __init__(self,data=None):
        self.data = data
        self.OPs = None
        self.OnlyOPs = None
        self.OPs_XyData = None  
        self.time0, self.time1, self.time2, self.time3 = None, None, None, None

    def set_operations(self, SL=60, TP_multiplier=0.3, start_timeH=7, start_timeM=2, delta_hour=4, reverse=False):
        """
        Abre operaciones si un FVG (fair value gap) es tocado en las próximas 6 velas:
        - Solo se abre dentro del horario [start_timeH:start_timeM, start_timeH:start_timeM + delta_hour].
        - FVG alcista (fvg=1): se abre compra al tocar fvg_price con el 'low' de la vela.
        - FVG bajista (fvg=-1): se abre venta al tocar fvg_price con el 'high' de la vela.
        - SL (Stop Loss) fijo en pips (SL).
        - TP (Take Profit) es (SL * TP_multiplier).
        - Se agrega 1 sola operación por cada FVG detectado.
        """

        # 1) Copiamos el DataFrame original
        df = self.data.copy()

        # 2) Creamos / reseteamos columnas donde guardaremos la operación
        df['OP'] = 0              # 0: no se abre operación; 1: buy; -1: sell
        df['SL'] = np.nan         # Precio de Stop Loss
        df['TP'] = np.nan         # Precio de Take Profit
        df['TP_Multiplier'] = np.nan #TP_multiplier
        df['open_price'] = np.nan # Precio de apertura de la operación

        # 3) Función de horas (NO tocar)
        def parametrizar_times(start_time1=time(start_timeH, start_timeM),
                            delta_hour=delta_hour):
            base_date = datetime.datetime(2000, 1, 1)
            time1_dt = datetime.datetime.combine(base_date, start_time1)
            time0_dt = time1_dt - timedelta(hours=1)
            time2_dt = time1_dt + timedelta(hours=delta_hour)
            time3_dt = time2_dt + timedelta(hours=1)
            return (time0_dt.time(), time1_dt.time(), time2_dt.time(), time3_dt.time())

        # Guardamos las horas en la instancia
        self.time0, self.time1, self.time2, self.time3 = parametrizar_times()

        # 4) Generamos (o conservamos) columnas auxiliares de fecha y hora
        #    (Si tu DataFrame ya tenía estas columnas, omite volver a crearlas)
        df['Date'] = df.index.date
        df['Time'] = df.index.time

        # 5) Parámetros de SL / TP
        #    Aquí pip_size = 1 para que "60 pips" equivalga a restar o sumar 60 al precio.
        #    Ajusta pip_size si tu instrumento usa otra escala.
        sl_pips = SL
        pip_size = 1

        # 6) Para cada fila que tenga un FVG (fvg != 0), buscamos en las 20 velas siguientes:
        #      - si se cumple la condición de "toque" (según fvg=1 o fvg=-1)
        #      - si está dentro del rango horario
        #    De cumplirse, marcamos la operación (OP, open_price, SL, TP) y no revisamos más velas.
        n_future_bars = 6
        last_index = len(df) - 1

        for i in range(len(df)):
            # Solo procesamos si en esta fila hay un FVG
            fvg_dir = df.at[df.index[i], 'fvg']  # 1 => alcista, -1 => bajista, 0 => no hay
            # Verificamos que esté dentro del horario permitido
            if not (self.time1 <= df.at[df.index[i], 'Time'] <= self.time2):
                continue
            if fvg_dir == 0:
                continue

            # Datos de este FVG
            fvg_price = df.at[df.index[i], 'fvg_price']

            # Recorremos hasta 20 velas siguientes (o hasta el final) para buscar el primer "toque"
            end_i = min(i + n_future_bars, last_index)
            # i+1 porque no abrimos operación en la misma vela que define el FVG, sino en alguna posterior
            for j in range(i + 1, end_i + 1):

                if fvg_dir == 1:
                    # FVG alcista: se abre BUY si el low toca o rompe fvg_price
                    if df.at[df.index[j], 'low'] <= fvg_price:
                        # Registramos la operación
                        df.at[df.index[j], 'OP'] = 1 if not reverse else -1
                        df.at[df.index[j], 'open_price'] = fvg_price

                        # SL por debajo del precio de apertura
                        sl_points = sl_pips * pip_size
                        df.at[df.index[j], 'SL'] = sl_points if not reverse else sl_points*TP_multiplier #----------------------

                        # TP por arriba => (SL * TP_multiplier)
                        tp_price = fvg_price + sl_pips * pip_size * TP_multiplier
                        df.at[df.index[j], 'TP'] = tp_price
                        df.at[df.index[j], 'TP_Multiplier'] = TP_multiplier if not reverse else 1/TP_multiplier #----------------------

                        # Terminamos para este FVG
                        break

                elif fvg_dir == -1:
                    # FVG bajista: se abre SELL si el high toca o rompe fvg_price
                    if df.at[df.index[j], 'high'] >= fvg_price:
                        # Registramos la operación
                        df.at[df.index[j], 'OP'] = -1 if not reverse else 1
                        df.at[df.index[j], 'open_price'] = fvg_price

                        # SL por arriba del precio de apertura
                        sl_points =sl_pips * pip_size
                        df.at[df.index[j], 'SL'] = sl_points if not reverse else sl_points*TP_multiplier  #----------------------

                        # TP por abajo => (SL * TP_multiplier)
                        tp_price = fvg_price - sl_pips * pip_size * TP_multiplier
                        df.at[df.index[j], 'TP'] = tp_price
                        df.at[df.index[j], 'TP_Multiplier'] = TP_multiplier if not reverse else 1/TP_multiplier #----------------------

                        # Terminamos para este FVG
                        break
        # 7) Limpiamos columnas temporales y asignamos resultado
        # ---------------------------------------------------------
        df.drop(columns=['Date','Time'], inplace=True)
        # 8) Asignamos el DataFrame resultante a self.OPs
        self.OPs = df

    def get_results(self, BE):
        df = self.OPs.copy()

        # Inicializar columnas de resultados
        df['Result'] = 0
        df['time_result'] = pd.NaT
        df['pips_driven'] = 0.0

        # Filtrar operaciones abiertas
        operations = df[df['OP'] != 0]

        for idx, operation in operations.iterrows():
            OP = operation['OP']
            SL = operation['SL']
            TP_multiplier = operation['TP_Multiplier']
            TP = SL * TP_multiplier
            open_price = operation['open_price']

            # Configuración de Break Even
            BE_enable = BE.get('enable', False)
            set_multiplier = BE.get('set', None)
            BE_tp_multiplier = BE.get('BE_tp', None)

            # Definir precios de TP y SL
            if OP == 1:
                TP_price = open_price + TP
                SL_price = open_price - SL
                if BE_enable:
                    set_level = open_price + SL * set_multiplier
                    BE_tp_price = open_price + SL * BE_tp_multiplier
            else:
                TP_price = open_price - TP
                SL_price = open_price + SL
                if BE_enable:
                    set_level = open_price - SL * set_multiplier
                    BE_tp_price = open_price - SL * BE_tp_multiplier

            # Definir ventana de tiempo (3 horas después de la apertura)
            start_time = idx
            end_time = start_time + pd.Timedelta(hours=3)
            operation_data = df[(df.index >= start_time) & (df.index <= end_time)]

            # Variables de control
            result = -2
            time_result = end_time
            pips_driven = 0.0
            be_triggered = False

            for op_idx, op_candle in operation_data.iterrows():
                high = op_candle['high']
                low = op_candle['low']

                # Verificar si es la primera vela (vela de apertura)
                es_primera_vela = (op_idx == start_time)

                if OP == 1:
                    # Evaluar TP solo si no es la primera vela
                    if not es_primera_vela:
                        if not be_triggered:
                            if high >= TP_price:
                                result = 1
                                time_result = op_idx
                                pips_driven = TP
                                break
                    # Evaluar SL siempre, incluyendo la primera vela
                    if low <= SL_price:
                        if not be_triggered:
                            result = -1
                            time_result = op_idx
                            pips_driven = -SL
                            break
                        else:
                            result = 2
                            time_result = op_idx
                            pips_driven = SL_price - open_price
                            break
                    # Configuración de Break Even
                    if BE_enable and not be_triggered:
                        if not es_primera_vela and high >= set_level:
                            be_triggered = True
                            SL_price = BE_tp_price
                else:
                    # Para operaciones de venta (OP != 1)
                    # Evaluar TP solo si no es la primera vela
                    if not es_primera_vela:
                        if not be_triggered:
                            if low <= TP_price:
                                result = 1
                                time_result = op_idx
                                pips_driven = TP
                                break
                    # Evaluar SL siempre, incluyendo la primera vela
                    if high >= SL_price:
                        if not be_triggered:
                            result = -1
                            time_result = op_idx
                            pips_driven = -SL
                            break
                        else:
                            result = 2
                            time_result = op_idx
                            pips_driven = open_price - SL_price
                            break
                    # Configuración de Break Even
                    if BE_enable and not be_triggered:
                        if not es_primera_vela and low <= set_level:
                            be_triggered = True
                            SL_price = BE_tp_price

            else:
                # Si no se alcanzó TP, SL ni BE dentro de la ventana
                if not operation_data.empty:
                    end_price = operation_data.iloc[-1]['close']
                    pips_driven = (end_price - open_price) if OP == 1 else (open_price - end_price)
                    time_result = operation_data.index[-1]
                else:
                    pips_driven = 0.0
                    time_result = end_time

            # Actualizar el DataFrame original
            df.at[idx, 'Result'] = result
            df.at[idx, 'time_result'] = time_result
            df.at[idx, 'pips_driven'] = pips_driven

        # Asegurarse de que 'time_result' es de tipo datetime y que el índice es DatetimeIndex
        df['time_result'] = pd.to_datetime(df['time_result'])

        self.OPs = df
        self.OnlyOPs = self.OPs[self.OPs['OP'] != 0]


class StrategyMetrics:
    def __init__(self, df_filt, risk_per_trade=1, inicial_balance=10000):
        """
        Inicializa la clase StrategyMetrics.

        Parámetros:
        - df_filt: DataFrame filtrado con los resultados de las operaciones.
        - risk_per_trade: Porcentaje de riesgo por operación.
        - inicial_balance: Balance inicial de la cuenta.
        """
        self.OPs_results = df_filt.copy()
        self.risk_per_trade = risk_per_trade
        self.inicial_balance = inicial_balance

    def get_resume(self, optimizador=False):
        """
        Calcula las métricas de la estrategia y genera los gráficos correspondientes.
        """

        # Calcular el riesgo monetario por operación basado en el capital inicial
        risk_amount = self.inicial_balance * (self.risk_per_trade / 100)

        # Copiar los resultados de operaciones
        df_ops = self.OPs_results.copy()

        # Asegurarse de que 'TP_Multiplier' existe y rellenar NaN con 1
        df_ops['TP_Multiplier'] = df_ops.get('TP_Multiplier', 1).fillna(1)

        # Calcular 'profit' para cada operación
        df_ops['profit'] = 0.0

        # Para 'Result' == 1 (ganada)
        df_ops.loc[df_ops['Result'] == 1, 'profit'] = risk_amount * df_ops.loc[df_ops['Result'] == 1, 'TP_Multiplier']

        # Para 'Result' == -1 (pérdida)
        df_ops.loc[df_ops['Result'] == -1, 'profit'] = -risk_amount

        # Para 'Result' == 2 (break even)
        df_ops.loc[df_ops['Result'] == 2, 'profit'] = (df_ops.loc[df_ops['Result'] == 2, 'pips_driven'] / df_ops.loc[df_ops['Result'] == 2, 'SL']) * risk_amount

        # Para 'Result' == -2 (operación cerrada por tiempo límite)
        df_ops.loc[df_ops['Result'] == -2, 'profit'] = (df_ops.loc[df_ops['Result'] == -2, 'pips_driven'] / df_ops.loc[df_ops['Result'] == -2, 'SL']) * risk_amount

        # Calcular el balance acumulado
        df_ops['Balance'] = self.inicial_balance + df_ops['profit'].cumsum()

        # Calcular el Drawdown Máximo
        df_ops['Pico'] = df_ops['Balance'].cummax()
        df_ops['Drawdown'] = (df_ops['Balance'] - df_ops['Pico']) / df_ops['Pico']
        max_drawdown = df_ops['Drawdown'].min() * 100  # Convertir a porcentaje

        # Calcular el Drawdown Técnico
        dd_counter = 0  # Contador de drawdown técnico
        technical_drawdown = []

        for idx, row in df_ops.iterrows():
            if row['Result'] == 1:
                # Ganancia: incrementar dd_counter por TP_Multiplier
                dd_counter += row.get('TP_Multiplier', 1)
            elif row['Result'] == -1:
                # Pérdida: disminuir dd_counter en 1, pero no menos de cero
                dd_counter = max(dd_counter - 1, 0)
            elif row['Result'] == 2:
                # BE no afecta el drawdown técnico
                pass
            elif row['Result'] == -2:
                # Operación cerrada por tiempo
                multiplier = row.get('pips_driven', 0) / row.get('SL', 1)
                if row['profit'] > 0:
                    dd_counter += round(multiplier, 1)
                else:
                    dd_counter = max(dd_counter - 1, 0)
            else:
                pass
            technical_drawdown.append(dd_counter)

        df_ops['Technical_Drawdown'] = technical_drawdown

        # Calcular el Drawdown Técnico Máximo
        df_ops['Pico_Tecnico'] = df_ops['Technical_Drawdown'].cummax()
        df_ops['Drawdown_Tecnico'] = df_ops['Pico_Tecnico'] - df_ops['Technical_Drawdown']
        max_technical_drawdown = df_ops['Drawdown_Tecnico'].max()

        # Calcular el Drawdown Recovery Time
        df_ops['Is_Drawdown'] = df_ops['Balance'] < df_ops['Pico']
        drawdown_periods = []
        recovery_times = []
        drawdown_start = None

        for idx, row in df_ops.iterrows():
            if row['Is_Drawdown']:
                if drawdown_start is None:
                    drawdown_start = idx
            else:
                if drawdown_start is not None:
                    recovery_time = (idx - drawdown_start).days
                    recovery_times.append(recovery_time)
                    drawdown_start = None

        # Si el último periodo está en drawdown y no se ha recuperado
        if drawdown_start is not None:
            recovery_time = (df_ops.index[-1] - drawdown_start).days
            recovery_times.append(recovery_time)

        if recovery_times:
            max_recovery_time = max(recovery_times)
        else:
            max_recovery_time = 0  # No hubo drawdowns

        # Calcular VaR (Value at Risk) al 95% de confianza
        df_profits = df_ops[df_ops['Result'].isin([1, -1])].copy()
        # Resamplear a periodos semanales y sumar los profits
        profits_periodic = df_profits['profit'].resample('ME').sum()
        returns = profits_periodic / self.inicial_balance
        var_percentile = 5  # 100% - 95% de confianza
        var = np.percentile(returns, var_percentile)
        var_monetary = var * self.inicial_balance

        # Calcular la expectativa matemática (solo Result == 1 y -1)
        df_expectation = df_ops[df_ops['Result'].isin([1, -1])]
        total_trades = len(df_expectation)
        win_trades = df_expectation[df_expectation['Result'] == 1].shape[0]
        loss_trades = df_expectation[df_expectation['Result'] == -1].shape[0]
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        loss_rate = loss_trades / total_trades if total_trades > 0 else 0

        average_win = df_expectation[df_expectation['profit'] > 0]['profit'].mean() if win_trades > 0 else 0
        average_loss = df_expectation[df_expectation['profit'] < 0]['profit'].mean() if loss_trades > 0 else 0

        expectation = (win_rate * average_win) + (loss_rate * average_loss)

        # Calcular ratios de Sortino y Sharpe
        profits_df = df_expectation[['profit']].copy()
        profits_df['Fecha'] = df_expectation.index
        profits_df.set_index('Fecha', inplace=True)

        # Resamplear a periodos semanales ('W') o mensuales ('ME')
        profits_periodic = profits_df['profit'].resample('W').sum()
        returns = profits_periodic / self.inicial_balance

        average_return = returns.mean()
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std(ddof=0)
        std_deviation = returns.std(ddof=0)

        sortino_ratio = average_return / downside_deviation if downside_deviation != 0 else np.nan
        sharpe_ratio = average_return / std_deviation if std_deviation != 0 else np.nan

        if not optimizador:
            # Preparar el resumen para imprimir
            resumen = {
                'Capital Inicial': f"${self.inicial_balance:,.2f}",
                'Capital Final': f"${df_ops['Balance'].iloc[-1]:,.2f}",
                'Drawdown Máximo': f"{max_drawdown:.2f}%",
                'Drawdown Técnico Máximo': f"{max_technical_drawdown}",
                'Drawdown Recovery Time': f"{max_recovery_time} días",
                'Operaciones Ganadas': win_trades,
                'Operaciones Perdidas': loss_trades,
                'Operaciones Timelimit': f"{(df_ops[df_ops['Result'] == -2].shape[0])}:  ${df_ops[df_ops['Result'] == -2]['profit'][df_ops['profit'] > 0].sum():,.2f} / ${df_ops[df_ops['Result'] == -2]['profit'][df_ops['profit'] <= 0].sum():,.2f}",
                'Operaciones BE': f"{df_ops[df_ops['Result'] == 2].shape[0]}: ${df_ops[df_ops['Result'] == 2]['profit'].sum():,.2f}",
                'Total de Operaciones': df_ops.shape[0],
                'Win Rate': f"{win_rate:.2%}",
                'VaR al 95% (ME)': f"${var_monetary:,.2f}",
                'Ratio de Sortino': f"{sortino_ratio:.4f}",
                'Ratio de Sharpe': f"{sharpe_ratio:.4f}",
                'Expectativa Matemática': f"${expectation:,.2f} por operación"
            }

            # Imprimir el resumen de manera estructurada
            print("\n" + "=" * 50)
            print("Resumen de la Estrategia".center(50))
            print("=" * 50)
            for clave, valor in resumen.items():
                print(f"{clave:<25}: {valor:>20}")
            print("=" * 50 + "\n")

            # Configurar la gráfica principal
            fig = plt.figure(figsize=(18, 36))
            gs = fig.add_gridspec(9, 2)

            # Subgráfico 1: Evolución del Capital
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(df_ops.index, df_ops['Balance'], marker='o', linestyle='-')
            ax1.set_title('Evolución del Capital')
            ax1.set_xlabel('Fecha')
            ax1.set_ylabel('Balance')
            ax1.grid(True)

            # Subgráfico 2: Drawdown
            ax2 = fig.add_subplot(gs[1, :])
            ax2.fill_between(df_ops.index, df_ops['Drawdown'] * 100, color='red', alpha=0.3)
            ax2.set_title('Drawdown a lo Largo del Tiempo')
            ax2.set_xlabel('Fecha')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True)

            # Subgráfico 3: Curva del Drawdown Técnico
            ax3 = fig.add_subplot(gs[2, :])
            ax3.fill_between(df_ops.index, -df_ops['Drawdown_Tecnico'], color='orange', alpha=0.3)
            ax3.set_title('Drawdown Técnico a lo Largo del Tiempo')
            ax3.set_xlabel('Fecha')
            ax3.set_ylabel('Drawdown Técnico (Puntos)')
            ax3.grid(True)

            # Subgráfico 4: Profit por Mes
            ax_new1 = fig.add_subplot(gs[3, :])
            profits_df_month = df_ops[['profit']].copy()
            profits_df_month['Month'] = profits_df_month.index.to_period('M').astype(str)
            profits_by_month = profits_df_month.groupby('Month')['profit'].sum()
            profits_by_month.plot(kind='bar', ax=ax_new1, color='skyblue')
            ax_new1.set_title('Profit por Mes')
            ax_new1.set_xlabel('Mes')
            ax_new1.set_ylabel('Profit')
            ax_new1.grid(True)
            # Puede continuar agregando los subgráficos adicionales según sea necesario

            plt.tight_layout()
            plt.show()
        else:
            return df_ops['Balance'].iloc[-1]#expectation *1000 # Ajuste según sea necesario
