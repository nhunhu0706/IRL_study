import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode = "rbg_array")

model = PPO("MlpPolicy", env, verbose=1, seed= 123, tensorboard_log="./ppo_cartpole_logs/")

model.learn(total_timesteps=95000)

model.save('pp0_cartpole_baseline')

obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic= True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()