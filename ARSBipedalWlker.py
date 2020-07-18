
import os
import gym
import numpy as np
from gym import wrappers

episode_length = 2000
lr = 0.02
num_dirs = 16
num_dirs_best = 16
noise = 0.03


class Normalizer:
    def __init__(self,n_inputs):
        self.n = np.zeros(n_inputs)
        self.mean = np.zeros(n_inputs)
        self.mean_diff = np.zeros(n_inputs)
        self.var = np.zeros(n_inputs)
        
    def observe(self,state):
        self.n+=1
        prev_mean = self.mean.copy()
        self.mean += (state - self.mean)/self.n
        self.mean_diff += (state-prev_mean)*(state-self.mean) 
        self.var = (self.mean_diff/self.n).clip(min=1e-2)
        
    def normalize(self,state):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (state - obs_mean)/obs_std
    
class Policy:
    def __init__(self,n_obs,n_actions,lr,n_dirs,n_dirs_best,noise):
        self.theta = np.zeros((n_actions,n_obs))
        self.learning_rate = lr
        self.num_directions = n_dirs
        self.num_best_directions = n_dirs_best
        self.noise = noise
        
    def evaluate(self,state,delta= None,direction = None):
        if direction is None:
            return self.theta.dot(state)
        elif direction == "positive":
            return (self.theta+self.noise*delta).dot(state)
        else:
            return (self.theta-self.noise*delta).dot(state)
        
    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for i in range(self.num_directions)]
        
    def update(self,rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for positive_reward,negative_reward,d in rollouts:
            step += (positive_reward-negative_reward)*d
        self.theta += self.learning_rate/(self.num_best_directions * sigma_r)*step
        
        
def play_one(env,normalizer,policy,direction = None,delta = None):
    state = env.reset()
    done = False
    total_reward = 0
    n_steps = 0.
    
    while not done and n_steps<episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1)
        total_reward+=reward
        n_steps+=1
    return total_reward

def train(env,policy,normalizer,epochs,lr,n_dirs,n_dirs_best,noise = 0.03):
    for epoch in range(epochs):
        deltas = policy.sample_deltas()
        positive_rewards = [0] * n_dirs
        negative_rewards = [0] * n_dirs
        
        for i in range(n_dirs):
            positive_rewards[i] = play_one(env,normalizer,policy,direction="positive",delta=deltas[i])
        for i in range(n_dirs):
            negative_rewards[i] = play_one(env,normalizer,policy,direction="negative",delta=deltas[i])
            
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()
        scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:n_dirs_best]
        rollout = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        policy.update(rollout, sigma_r)
        policy_eval = play_one(env,normalizer,policy)

        print('Epoch:', epoch, 'Reward:', policy_eval)

if __name__ == '__main__':
    np.random.seed(1)
    env = gym.make('BipedalWalker-v2')
    env = gym.wrappers.Monitor(env,'Episode',force=True)   
    policy = Policy(env.observation_space.shape[0],env.action_space.shape[0],lr,num_dirs,num_dirs_best,noise)
    normalizer = Normalizer(env.observation_space.shape[0])
    train(env,policy,normalizer,1000,lr,num_dirs,num_dirs_best,noise)
    


    
    
        
    
        
        
