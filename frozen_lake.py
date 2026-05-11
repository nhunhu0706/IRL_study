import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
def run(episodes):
    SEED = 42
    env = gym.make('FrozenLake-v1', map_name = '4x4', is_slippery=True, render_mode = None)
    np.random.seed(SEED)
    env.action_space.seed(SEED)
    q = np.zeros((env.observation_space.n, env.action_space.n))

    a = 0.1 #learning rate alpha
    g = 0.9 #discount factor gamma
    e = 1
    e_decay_rate = 2/episodes
    rng = np.random.default_rng(SEED)
    rewards_per_episode=np.zeros(episodes)

    for i in range(episodes):
        if i == 0:
            state, _ = env.reset(seed=SEED)
        else:
            state, _ = env.reset()
        terminated = False
        truncated = False
        while(not truncated and not terminated):
            if rng.random() < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])
            new_state, reward, terminated, truncated, _ = env.step(action)
            q[state, action] = q[state, action] + a*(reward + g* np.max(q[new_state,:])-q[state,action])

            state = new_state
        e = max(e - e_decay_rate,0)   
        if reward == 1:
            rewards_per_episode[i] = 1
    sum_rewards = np.zeros(episodes)
    env.close()
    
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0,t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.show()
    f= open('frozen_lake.pkl','wb')
    pickle.dump(q,f)
    f.close()

env = gym.make('FrozenLake-v1', map_name = '4x4', is_slippery=True, render_mode = None)
with open('frozen_lake.pkl', 'rb') as f:
    q = pickle.load(f)
def test_agent(episodes=100):
    wins = 0
    SEED = 42

    for i in range(episodes):
        if i == 0:
            state, _ = env.reset(seed=SEED)
        else:
            state, _ = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = np.argmax(q[state, :])
            
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated and reward == 1.0:
                wins += 1
    env.close()
    print(wins)

if __name__ == '__main__':
    run(15000)
    test_agent()
