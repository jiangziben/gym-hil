import matplotlib.pyplot as plt
import gymnasium as gym
import gym_hil  # noqa: F401
import numpy as np
# import charging_task_gym_env  # noqa: F401
env = gym.make(id="gym_hil/ChargingTaskGamepad-v0",render_mode="human")
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
while not done:
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    # plt.imshow(obs["pixels"]["front"])
    # plt.pause(0.01)
    # done = terminated or truncated
env.close()