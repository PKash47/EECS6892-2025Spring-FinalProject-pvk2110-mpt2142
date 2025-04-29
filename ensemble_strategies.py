import os
import gymnasium as gym
import finrl
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def make_env(env_kwargs):
    def _init():
        env = StockTradingEnv(
            df=env_kwargs['df'],
            stock_dim=env_kwargs['stock_dim'],
            hmax=env_kwargs['hmax'],
            initial_amount=env_kwargs['initial_amount'],
            num_stock_shares = env_kwargs['num_stock_shares'],
            buy_cost_pct=env_kwargs['buy_cost_pct'],
            sell_cost_pct=env_kwargs['sell_cost_pct'],
            reward_scaling=env_kwargs['reward_scaling'],
            state_space=env_kwargs['state_space'],
            action_space=env_kwargs['action_space'],
            tech_indicator_list=env_kwargs['tech_indicator_list'],
            turbulence_threshold=env_kwargs['turbulence_threshold']
        )
        return env
    return _init

def run_ensemble_strategy(data: pd.DataFrame, models_list: list, stock, total_timesteps:  int = 10000, save_dir="trained_models"):
    # stock_dimension = len(data.tic.unique())
    stock_dimension = data['tic'].nunique()
    # num_stock_shares = [0] * len(data['tic'].unique())# [0] #* stock_dimension
    num_stock_shares = [0] * data['tic'].nunique()
    INDICATORS = ["macd", "rsi_30", "cci_30", "adx_30"]

    env_kwargs = {
        "df": data,
        "stock_dim": stock_dimension,
        "hmax": 100,
        "initial_amount": 1e6,
        "num_stock_shares": num_stock_shares,
        'buy_cost_pct': [0.001] * data['tic'].nunique(),
        'sell_cost_pct': [0.001] * data['tic'].nunique(),
        "reward_scaling": 1e-4,
        "state_space": 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension,
        "action_space": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "turbulence_threshold": 250
    }

    env_kwargs['num_stock_shares'] = [0] * env_kwargs['stock_dim']

    os.makedirs(save_dir, exist_ok=True)
    env = DummyVecEnv([make_env(env_kwargs)])

    trained_models = {}

    for model_name in models_list:
        print(f"\nTraining {stock} {model_name} model\n")

        if model_name == "PPO":
            model = PPO('MlpPolicy', env, verbose=0)
        elif model_name == "A2C":
            model = A2C('MlpPolicy', env, verbose=0)
        elif model_name == "DDPG":
            model = DDPG('MlpPolicy', env, verbose=0)
        else:
            raise ValueError(f"Model {model_name} not supported!")

        model.learn(total_timesteps=total_timesteps)
        trained_models[model_name] = model

        model.save(os.path.join(save_dir, f"{stock}_{model_name}_model"))

        print(f"{stock} {model_name} training completed and saved.\n")

    return trained_models


def ensemble_predict(models_dict, obs):
    """
    Take the action that the model ensemble averages.
    """
    actions = []
    for model in models_dict.values():
        action, _ = model.predict(obs) # need this deterministic=True?
        actions.append(action)

    # Majority vote or Average action?
    actions_array = np.array(actions)
    final_action = np.mean(actions_array, axis=0) # need to round? np.round(np.mean(actions_array, axis=0))

    return final_action

def run_ensemble_trading(models_dict, env, n_steps=1000):
    """
    Use ensemble strategy to trade for n_steps and track portfolio value.
    """
    obs = env.reset()
    portfolio_values = []
    total_reward = 0
    for step in range(n_steps):
        action = ensemble_predict(models_dict, obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        # Save the current portfolio value
        if hasattr(env, "envs"):  # DummyVecEnv
            portfolio_value = env.envs[0].asset_memory[-1]
        else:  # Normal env
            portfolio_value = env.asset_memory[-1]

        portfolio_values.append(portfolio_value)

        if done:
            break


    print(f"Ensemble strategy total reward: {total_reward}")
    return portfolio_values


