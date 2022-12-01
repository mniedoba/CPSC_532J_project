import copy
import gym
import numpy as np
from scipy.special import logsumexp
from tqdm import trange

import torch

# Todo:
#  - Implement SMC with some policy \theta
#


def resample(particles, weights):
    """Resample a set of particles based on weights."""
    n_particles = len(particles)
    step_size = 1 / n_particles
    cur_weight = np.random.random() / n_particles  # Start with a random cumulative weight between 0 and 1/N.
    cumulative_weights = np.cumsum(weights)
    new_particles = []
    particle_idx = 0
    for i in range(n_particles):
        while cur_weight > cumulative_weights[particle_idx]:
            particle_idx += 1
        new_particle = copy.deepcopy(particles[particle_idx])
        new_particle.logW = 0
        new_particles.append(new_particle)
        cur_weight += step_size
    assert(len(particles) == len(new_particles))
    return new_particles



class RolloutParticle:

    def __init__(self, env, state):
        self.env = env
        self.state = state
        self.done=False
        self.logW = 0.


def SMC(model, n_particles, env_id: str, seeds, scale=0.01):
    """Run SMC on an environment given a policy.

    Args:
        Policy: Type? The policy network to produce action predictions.
        n_particles: An integer representing the number of particles to run
        env: A str environment key.
        seed: An integer specifying the starting random seed for the environment.
    """

    particles = []
    for i in range(n_particles):
        env = gym.make(env_id)
        env.seed(int(seeds[i]))
        obs = env.reset()
        particles.append(RolloutParticle(env, obs))

    done = False
    log_evidence = 0
    step = 0
    log_total_weights = 0
    while not done:
        done = True
        for i, particle in enumerate(particles):
            if particle.done:
                continue
            actions = model.predict(particle.state, deterministic=False)
            obs, reward, env_done, _ = particle.env.step(actions[0])
            particle.state = obs
            particle.logW += reward * scale
            particle.done = env_done
            done &= particle.done
        unormalized_log_weights = np.array([particle.logW for particle in particles])
        log_total_weights = logsumexp(unormalized_log_weights)
        normalized_log_weights = unormalized_log_weights - logsumexp(unormalized_log_weights)

        normalized_weights = np.exp(normalized_log_weights)
        ESS = 1. / (normalized_weights ** 2).sum()
        # print(f'ESS: {ESS/n_particles}')
        if ESS < 0.5 * n_particles:
            # print("Resampling")
            log_evidence += -np.log(n_particles) + log_total_weights
            particles = resample(particles, np.exp(normalized_log_weights))
        step += 1
    log_evidence += -np.log(n_particles) + log_total_weights
    return log_evidence
