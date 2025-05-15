import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.env_checker import check_env

import matplotlib.pyplot as plt
from Utils.TradingStation_v01 import TradingEnvironment

def make_env_test():
    # Aquí instancias tu entorno real, por ejemplo, conectado a la API o con render habilitado
    # Supongamos que 'TradingEnvironment' es capaz de recibir render_mode='human'
    env = TradingEnvironment(
        csv_path='US30.cash_data_m1_antiguo.csv',  # o la configuración de tu API
        render_mode=None,  # habilita el render para visualizar
        sl=30,
        tp=40,
        balance=10000,
        risk=240
    )
    # Aplica el flattening para transformar observaciones en vectores
    env = FlattenObservation(env)
    # Usa el Monitor para guardar estadísticas
    env = Monitor(env)
    return env

env_test = DummyVecEnv([make_env_test])
env_test = VecNormalize.load("vec_normalize.pkl", env_test)
env_test.training = False
env_test.norm_reward = False
model = PPO.load("ppo_trading_model", env=env_test,device="cpu")

# Ejemplo de evaluación: ejecuta el agente en el entorno
win,loss,count=0,0,0
obs = env_test.reset()  # Se hace el reset inicial
while count<700:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env_test.step(action)
    if dones:
        if rewards==15:
            win+=1
        else:
            loss+=1
        count+=1
    # Renderiza el entorno (si tu render está implementado)
    #env_test.render()
print(f"ratio:{win/(win+loss)}, Win: {win}, Loss: {loss}")