# Goal: Create plots of SMC estimates of performance.
# Use the same seeds as in the baseline_episodes.py
# Random choice n seeds, perform SMC, report log(P(O)).
# Repeat N times, create the same kind of plot.
import os

import numpy as np
from stable_baselines3 import SAC
from tqdm import trange, tqdm

from SMC import SMC

ENTROPY = 109823573

def collect_smc_runs(n_runs, particles_per_run, scale):
    policy_fmt_path = '../rl-baselines3-zoo/logs/sac/Pendulum-v1_11/rl_model_{0}_steps'
    out_folder = f'smc_results/{particles_per_run}p'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    seeds = np.random.SeedSequence(ENTROPY).generate_state(1000)

    for step in trange(2000, 3001, 500, desc='Model Step'):
        out_path = os.path.join(out_folder, f'step_{step}_scale_{scale}.npy')
        # if os.path.exists(out_path):
            # continue
        results = np.zeros(n_runs)
        model = SAC.load(f'../rl-baselines3-zoo/logs/sac/Pendulum-v1_11/rl_model_{step}_steps')
        for i in trange(n_runs, leave=False, desc='SMC Runs'):
            run_seeds = np.random.choice(seeds, particles_per_run)
            results[i] = SMC(model, particles_per_run, env_id='Pendulum-v1', seeds=run_seeds, scale=scale)
        # np.save(out_path, results)


if __name__ == '__main__':
    n_runs=1
    n_particles=3
    scale=0.01
    for n_particles in (pbar := tqdm([50])):
        pbar.set_description(f'SMC for {n_particles} Particles')
        for scale in [0.01]:
            pbar.set_postfix({'Scale': scale})
            collect_smc_runs(n_runs, n_particles, scale)