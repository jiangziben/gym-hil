import matplotlib.pyplot as plt
import gymnasium as gym
import charging_task_gym_env  # noqa: F401
env = gym.make(id="ChargingTask-v0",render_mode="human",reward_mode="sparse")
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
while not done:
    # action = env.action_space.sample()
    # obs, reward, terminated, truncated, info = env.step(action)
    # done = terminated or truncated
    env.render()  # a display is required to render
env.close()