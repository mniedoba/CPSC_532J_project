import os

import numpy as np
import matplotlib.pyplot as plt

# MODEL_REWARDS_FORMAT = './baseline_results/model_rewards_step_{0}.npy'
MODEL_REWARDS_FORMAT = './smc_results/50p/step_{0}_scale_0.001.npy'


def main(sample_size, n_samples):
    np.random.seed(0)
    mean, low, high = [], [], []
    steps = [step for step in range(500, 10001, 500)]
    for step in steps:
        reward_file = MODEL_REWARDS_FORMAT.format(step)
        if not os.path.exists(reward_file):
            continue
        rewards = np.load(reward_file)
        print(rewards.mean())
        reward_samples = np.zeros(n_samples)
        for i in range(n_samples):
            reward_samples[i] = np.random.choice(rewards, sample_size, replace=False).mean()
        mean.append(reward_samples.mean())
        # low.append(np.percentile(reward_samples, 2.5))
        # high.append(np.percentile(reward_samples, 97.5))
        low.append(np.percentile(rewards, 2.5))
        high.append(np.percentile(rewards, 97.5))
    n_elems = len(mean)
    plt.figure(figsize=(10,10))
    plt.plot(steps[:n_elems], mean, label='Sample Mean')
    plt.fill_between(steps[:n_elems], low, high, alpha=0.1, label='95% Interval')
    plt.legend(loc='lower right')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'Log P(O) with 10 particles')
    plt.xlim(500, steps[n_elems - 1])
    plt.ylim(-1.9, .05)
    # plt.savefig(f'./img/smc/5p.png')
    plt.show()

if __name__ == "__main__":
    sample_size = 1
    n_samples = 100
    main(sample_size, n_samples)