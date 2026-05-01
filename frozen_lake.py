import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes):
    env = gym.make('FrozenLake-v1', map_name = '4x4', is_slippery=False, render_mode = None)
    SEED = 42

    np.random.seed(SEED)
    q = np.zeros((env.observation_space.n, env.action_space.n))

    a = 0.1 #learning rate alpha
    g = 0.9 #discount factor gamma
    e = 1
    e_decay_rate = 1/episodes
    rng = np.random.default_rng()
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


env = gym.make('FrozenLake-v1', map_name = '4x4', is_slippery=False, render_mode = 'human')

with open('frozen_lake.pkl', 'rb') as f:
    q = pickle.load(f)
def test_agent(episodes=5):
    for i in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # 3. Policy: Luôn chọn hành động tốt nhất (Exploit) 
            # Đặt epsilon = 0 [cite: 139, 233]
            action = np.argmax(q[state, :])
            
            state, reward, terminated, truncated, _ = env.step(action)

    env.close()

if __name__ == '__main__':
    run(2000)
    test_agent()