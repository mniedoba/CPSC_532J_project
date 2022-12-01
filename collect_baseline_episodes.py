import os

import gym
from stable_baselines3 import SAC
import torch
from tqdm import trange

import numpy as np

ENTROPY = 109823573

def load_policy(filepath):
    model = SAC.load(filepath)
    return model

@torch.no_grad()
def collect_episode_reward(model, seed, viz=False):
    env = gym.make('Pendulum-v1')
    env.seed(int(seed))
    state = env.reset()
    done = False
    total_reward = 0.
    while not done:
        if viz:
            env.render()
        actions = model.predict(state, deterministic=False)
        state, reward, done, _ = env.step(actions[0])
        total_reward += reward
    env.close()
    return total_reward



def main():
    policy_fmt_path = '../rl-baselines3-zoo/logs/sac/Pendulum-v1_11/rl_model_{0}_steps'
    out_fmt_path = 'baseline_results/model_rewards_step_{0}.npy'
    n_episodes_per_step = 1000
    seeds = np.random.SeedSequence(109823573).generate_state(n_episodes_per_step)
    for step in trange(50000, 499, -500, desc='Step'):
        policy_path = policy_fmt_path.format(step)
        policy = load_policy(policy_path)
        out_path = out_fmt_path.format(step)
        if os.path.exists(out_path):
            continue
        episode_rewards = np.zeros(n_episodes_per_step)
        for i in trange(n_episodes_per_step, desc='Episode'):
            episode_rewards[i] = collect_episode_reward(policy, seeds[i])
        np.save(out_path, episode_rewards)



if __name__ == '__main__':
    main()