""" note:
pip install --upgrade numpy
pip3 install opencv-python """

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments (with alternatives) (Classic Control Environments)
vec_env = make_vec_env("CartPole-v1", n_envs=4)
#vec_env = make_vec_env("Pendulum-v1", n_envs=4)
#vec_env = make_vec_env("Acrobot-v1", n_envs=4)
# vec_env = make_vec_env("MountainCar-v0", n_envs=4)
#vec_env = make_vec_env("MountainCarContinuous-v0", n_envs=4)



model = PPO("MlpPolicy", vec_env, verbose=1) # no additional hyperparameter (and so no tuning)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")