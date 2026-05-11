import numpy as np
import pickle
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
def soft_policy_iteration(rewards, P, gamma = 0.9, threshold=1e-5):
    n_states, n_actions, _ = P.shape
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))

    while True:
        V_prev = np.copy(V)

        for s in range(n_states):
            for a in range(n_actions):

                expected_v = np.dot(P[s,a,:],V)
                Q[s,a] = rewards[s] + gamma * expected_v
        
        max_q = np.max(Q,axis=1)
        V = max_q + np.log(np.sum(np.exp(Q-max_q[:,np.newaxis]),axis=1))
        if np.max(np.abs(V-V_prev)) < threshold:
            break
    
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        policy[s,:] = np.exp(Q[s,:]-V[s])
    
    return policy

def expected_counts(policy, P, n_states, n_actions, n_step = 100):
    D = np.zeros((n_step, n_states))

    D[0,0] = 1.0

    for t in range(n_step - 1):
        for s in range(n_states):
            for a in range(n_actions):
                prob_s_a = D[t,s] * policy[s,a]

                for s_next in range(n_states):
                    D[t+1, s_next] += prob_s_a * P[s,a,s_next]

    expected_counts = np.sum(D, axis=0)
    return expected_counts

def expert_counts(q, P, n_states, n_actions, n_step = 100):
    policy = np.zeros((n_states,n_actions))

    for s in range(n_states):
        if np.max(q[s]) == 0 and np.min(q[s])==0:
            policy[s] = 1.0/n_actions
        else:
            best_action = np.argmax(q[s])
            policy[s,best_action] = 1
    
    expert_cnts = expected_counts(policy,P,n_states,n_actions,n_step)
    return expert_cnts

def maxent(q, P, n_states, n_actions, epochs, lr):
    rewards = np.zeros(n_states)

    expert_cnts = expert_counts(q, P, n_states, n_actions)

    for _ in range(epochs):
        policy = soft_policy_iteration(rewards,P)
        expected_cnts = expected_counts(policy,P,n_states,n_actions)
        grad = expert_cnts - expected_cnts
        rewards += lr * grad

    return rewards
with open('frozen_lake.pkl', 'rb') as f:
    q = pickle.load(f)
env = gym.make("FrozenLake-v1", is_slippery=True).unwrapped
n_states = env.observation_space.n
n_actions = env.action_space.n 
def get_P(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    P = np.zeros((n_states,n_actions,n_states))

    for s in range(n_states):
        for a in range(n_actions):
            for prob, next_s, reward, done in env.P[s][a]:
                P[s,a,next_s] +=prob
    
    return P
P = get_P(env)
irl_reward = maxent(q, P, n_states, n_actions, 1000, 0.1 )
def BC(q, n_states, n_actions):
    policy = np.zeros((n_states,n_actions))

    for s in range(n_states):
        if np.max(q[s]) == 0 and np.min(q[s])==0:
            policy[s] = 1.0/n_actions
        else:
            best_action = np.argmax(q[s])
            policy[s,best_action] = 1
    return policy
def test_agent(env, policy, episodes=100):
    wins =0
    for i in range(episodes):
        if i ==0:
            state, _ = env.reset(seed = 111)
        state, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = int(np.argmax(policy[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated and reward == 1.0:
                wins += 1
    return wins
def IRL_policy(reward, P, gamma=0.9, threshold=1e-5):
    n_states, n_actions,_ = P.shape
    V = np.zeros(n_states)
    Q = np.zeros((n_states,n_actions))

    while True:
        V_prev = np.copy(V)
        for s in range(n_states):
            for a in range(n_actions):
                expected_v = np.dot(P[s,a,:],V)
                Q[s,a] = reward[s] + gamma * expected_v
        V = np.max(Q, axis=1)

        if np.max(np.abs(V-V_prev)) < threshold:
            break
    policy = np.zeros((n_states,n_actions))
    for s in range(n_states):
        best_action = np.argmax(Q[s,:])
        policy[s,best_action]=1
    return policy
SEED = 42
random_map = generate_random_map(size=4, p=0.8, seed=SEED)
env_transfer = gym.make("FrozenLake-v1", desc=random_map, is_slippery=True).unwrapped
state, info = env_transfer.reset(seed=SEED)
P_transfer = get_P(env_transfer)
print(test_agent(env_transfer, BC(q,n_states,n_actions)))
print(test_agent(env_transfer,IRL_policy(irl_reward,P_transfer)))