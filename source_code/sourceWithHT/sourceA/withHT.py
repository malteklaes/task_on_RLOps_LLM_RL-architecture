""" note:
pip install --upgrade numpy
pip3 install opencv-python """

import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


vec_env = make_vec_env("CartPole-v1", n_envs=4)

# HYPERPARAMETER TUNING
learning_rates = [0.001, 1e-34, 0.9]

for learning_rate in learning_rates:
    print(f"Training model with learning rate: {learning_rate}")
    
    # apply hyperparameters
    model = PPO("MlpPolicy", vec_env, learning_rate=learning_rate, verbose=1)

    # train model
    model.learn(total_timesteps=25000)
    
    # save model
    model_name = f"ppo_cartpole_lr_{learning_rate}"
    model.save(model_name)
    
    del model

    model = PPO.load(model_name)

    obs = vec_env.reset()
    start_time = time.time()
    while (time.time() - start_time) < 2:  # animation for just 2 seconds
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")

