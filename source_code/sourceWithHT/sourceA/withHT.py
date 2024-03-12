""" note:
pip install --upgrade numpy
pip3 install opencv-python """

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

# HYPERPARAMETER TUNING
learning_rate = 0.001 # good between 1e-5 and 0.1
learning_rate_bad = 1e-34 # to low
learning_rate_veryGood = 0.9 # very high

# params: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.PPO
model = PPO("MlpPolicy", vec_env, learning_rate=learning_rate, verbose=1)

# trian model
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

# load saved model
model = PPO.load("ppo_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
