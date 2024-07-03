# !pip install gymnasium

import gymnasium as gym
import random
import numpy as np 

# Parameters
ENV_NAME = "FrozenLake-v1" 
GAMMA = 0.99
ALPHA = 0.1
MAX_EPSILON = 1
MIN_EPSILON = 0.05
DECAY_RATE = 0.005
TRAIN_EPISODES = 100
EVAL_EPISODES = 20

class Agent: 
  def __init__(self): 
    self.env = gym.make(ENV_NAME)
    self.states_n = self.env.observation_space.n
    self.actions_n = self.env.action_space.n 
    self.Qtable0 = np.zeros((self.states_n, self.actions_n))
  
  def greedy_policy(self, state, Qtable): 
    action = np.argmax(Qtable[state])
    return action 
  
  def epsilon_greedy(self, state, Qtable, epsilon): 
    random_num = np.random.uniform(0, 1) 
    if random_num < epsilon: 
      action = self.env.action_space.sample() # Explore
    else: 
      action = self.greedy_policy(state, Qtable) # Exploit
    return action
  
  def train(self): 
    Qtable = self.Qtable0
    for e in range(TRAIN_EPISODES):
      state, _ = self.env.reset()
      epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE*e)
      while True: 
        action = self.epsilon_greedy(state, Qtable, epsilon) 
        s_new, r, terminated, truncated, _ = self.env.step(action)

        # update Qtable
        Qtable[state][action] = Qtable[state][action] + ALPHA * (
            float(r) + GAMMA * np.max(Qtable[s_new]) - Qtable[state][action]
        )
        if terminated or truncated: 
          break 
        state = s_new
    return Qtable
  
  def eval(self, Qtable):
    r_episodes = [] # total reward in all episodes
    for e in range(EVAL_EPISODES): 
      state, _ = self.env.reset()
      r_episode = 0
      while True:
        action = self.greedy_policy(state, Qtable) 
        s_new, r, terminated, truncated, _ = self.env.step(action) 
        r_episode += float(r) 
        if terminated or truncated: 
          break 
        state = s_new 
      r_episodes.append(r_episode)
    mean_reward = np.mean(r_episodes) 
    std_reward = np.std(r_episodes) 
    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")


agent = Agent()
Qtable = agent.train()
agent.eval(Qtable)
