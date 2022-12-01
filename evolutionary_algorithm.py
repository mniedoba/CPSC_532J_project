# An implementation of evolutionary algorithms.
import copy
import os
import time

from tqdm import trange
import numpy as np
from scipy.special import logsumexp
import gym
import wandb

import torch
from torch.distributions import Normal
import torch.nn as nn

from SMC import SMC
# Todo:
# SMC
# Baseline GA

LOG_STD_MAX = 2
LOG_STD_MIN = -20
ENTROPY=52109271092

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

class SimpleActor(nn.Module):
    """Simple Stochastic Actor."""

    def __init__(self):
        super().__init__()
        self.latent_policy = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mu = nn.Linear(64, 1)
        self.log_std = nn.Linear(64, 1)

    def forward(self, obs):
        latent_pi = self.latent_policy(obs)
        mu = self.mu(latent_pi)
        log_std = self.log_std(latent_pi)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        action_dist = Normal(mu, torch.exp(log_std))
        gaussian_actions = action_dist.sample()
        # TODO: Replace 2 with the action space bounds.
        return torch.tanh(gaussian_actions) * 2

    def predict(self, obs, deterministic=True):
        action_tensor = self.forward(torch.tensor(obs))
        return action_tensor.numpy()[None, :]




class EvolutionaryAlgorithm:

    def __init__(self, population_size):
        self.population_size = population_size
        self.population = self.build_population()

    def build_population(self):
        """Build an initial population."""
        raise NotImplementedError()

    def fitness_function(self):
        """Evaluate the fitness of the population.

        Returns:
            A List of Fitness scores for each individual.
        """
        raise NotImplementedError()

    def evolve_population(self, fitness_scores):
        """Given a population of individuals and their respective fitness, evolve the next generation.

        Args:
            fitness_scores:    A list of N fitness scores corresponding to the individuals.

        """
        raise NotImplementedError()

    def train(self, n_generations, max_time=None):
        start = time.time()

        for _ in trange(n_generations):
            if max_time and time.time() > start + max_time:
                break

            fitness_scores = self.fitness_function()
            self.population = self.evolve_population(fitness_scores)

        return self.population


class SMCEvolutionaryAlgorithm(EvolutionaryAlgorithm):

    def __init__(self, cfg):
        super().__init__(cfg.n_policy_particles)
        self.population_log_weights = torch.zeros(cfg.n_policy_particles)
        self.n_particles = cfg.n_episode_particles
        self.gen = 0
        self.mutation_prob = cfg.mutation_prob
        self.n_eval_episodes = cfg.n_eval_samples
        self.n_train_repeats = cfg.n_train_repeats
        self.reward_scale = cfg.reward_scale

        self.param_noise_std = cfg.param_noise_std
        self.param_noise_prob = cfg.param_noise_prob

        self.viz = cfg.viz
        self.wandb = cfg.wandb
        self.ckpt = cfg.save_ckpt

        self.seed = cfg.seed

    def build_population(self):
        population = []
        for _ in range(self.population_size):
            population.append(SimpleActor())
        return population

    def fitness_function(self):
        fitness_scores = np.zeros(self.population_size)
        seeds = np.random.SeedSequence(ENTROPY + self.gen + self.seed*1000).generate_state(self.n_particles)
        # Todo: add multiprocessing.
        for i, agent in enumerate(self.population):
            log_p_optim = SMC(agent, n_particles=self.n_particles, env_id='Pendulum-v1', seeds=seeds, scale=self.reward_scale)
            fitness_scores[i] += log_p_optim
        return fitness_scores

    def eval_agent(self, agent):
        prev_total_reward = 0
        seeds = np.random.SeedSequence(ENTROPY + self.gen + self.seed*1000).generate_state(self.n_particles)
        for i in range(self.n_particles):
            viz = self.viz and i == 0
            for j in range(self.n_train_repeats):
                prev_total_reward += collect_episode_reward(agent, seeds[i], viz=viz) / self.n_train_repeats

        eval_total_reward = 0
        seeds = np.random.SeedSequence(ENTROPY - self.gen).generate_state(self.n_eval_episodes)
        for i in range(self.n_eval_episodes):
            viz = self.viz and i == 0
            eval_total_reward += collect_episode_reward(agent, seeds[i], viz=viz)
        if self.ckpt:
            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')
            torch.save(agent, f'checkpoints/{self.gen}.pt')
        if self.wandb:
            wandb.log({'train_mean_reward': prev_total_reward / self.n_particles,
                       'eval_reward': eval_total_reward / self.n_eval_episodes}, step=self.gen)
        else:
            print('Train reward:', prev_total_reward / self.n_particles,
                  'Eval Reward:', eval_total_reward / self.n_eval_episodes)

    @torch.no_grad()
    def evolve_population(self, fitness_scores):
        normalized_log_weights = fitness_scores - logsumexp(fitness_scores) # log(exp(fitness_score) / exp(fitness_score).sum())
        new_population = []

        normalized_weights = np.exp(normalized_log_weights)
        ESS = 1. / (normalized_weights ** 2).sum()

        if self.wandb:
            wandb.log({
                'ESS': ESS,
                'Fractional SS': ESS/ self.population_size,
                'Max Log Weight': fitness_scores.max() / self.reward_scale
            }, step=self.gen)
        else:
            print(f'Generation {self.gen }. ESS: {ESS}, Fractional Sample Size: {ESS / self.population_size}, Max log weight: {fitness_scores.max() / self.reward_scale}')
        self.eval_agent(self.population[np.argmax(fitness_scores)])

        # Systematic Resampling
        step_size = 1. / self.population_size
        cur_weight = np.random.random() / self.population_size
        cumulative_weights = np.cumsum(np.exp(normalized_log_weights))
        particle_idx = 0
        for i in range(self.population_size):
            while cur_weight > cumulative_weights[particle_idx]:
                # print(cur_weight, particle_idx, cumulative_weights[particle_idx])
                particle_idx += 1
            new_agent = copy.deepcopy(self.population[particle_idx])
            for param in new_agent.parameters():
                if not param.requires_grad:
                    continue
                new_param = param

                # Add random noise centered on current parameter values.
                noise_mask = torch.randn_like(new_param) < self.param_noise_prob
                new_param = new_param + self.param_noise_std * torch.randn_like(new_param) * noise_mask

                # Reset some parameters to 0 mean unit variance samples.
                mutation_mask = torch.rand_like(new_param)
                new_value = torch.where(mutation_mask < self.mutation_prob, torch.randn_like(new_param), new_param)
                param.copy_(new_value)
            new_population.append(new_agent)
            cur_weight += step_size
        self.gen += 1
        return new_population


class TopKEvolutionaryAlgorithm(EvolutionaryAlgorithm):

    def __init__(self, cfg):
        super().__init__(cfg.n_policy_particles)
        self.n_trials = cfg.n_episode_particles
        self.k = int(cfg.n_policy_particles * cfg.keep_fraction)
        self.keep_elites = cfg.keep_elites
        self.gen = 0

        self.param_noise_std = cfg.param_noise_std
        self.param_noise_prob = cfg.param_noise_prob
        self.mutation_prob = cfg.mutation_prob

        self.n_eval_episodes = cfg.n_eval_samples
        self.n_train_repeats = cfg.n_train_repeats

        self.wandb = cfg.wandb
        self.viz = cfg.viz
        self.seed = cfg.seed

    def build_population(self):
        population = []
        for _ in range(self.population_size):
            population.append(SimpleActor())
        return population

    def eval_agent(self, agent):
        prev_total_reward = 0
        seeds = np.random.SeedSequence(ENTROPY + self.gen + self.seed*1000).generate_state(self.n_trials)
        for i in range(self.n_trials):
            viz = self.viz and i == 0
            for j in range(self.n_trials):
                prev_total_reward += collect_episode_reward(agent, seeds[i], viz=viz) / self.n_train_repeats

        eval_total_reward = 0
        seeds = np.random.SeedSequence(ENTROPY - self.gen).generate_state(self.n_eval_episodes)
        for i in range(self.n_eval_episodes):
            viz = self.viz and i == 0
            eval_total_reward += collect_episode_reward(agent, seeds[i], viz=viz)
        if self.wandb:
            wandb.log({'topk/train_mean_reward': prev_total_reward / self.n_trials,
                       'topk/eval_reward': eval_total_reward / self.n_eval_episodes}, step=self.gen)
        else:
            print('Train reward:', prev_total_reward / self.n_trials,
                  'Eval Reward:', eval_total_reward / self.n_eval_episodes)

    def fitness_function(self):
        fitness_scores = np.zeros(self.population_size)
        seeds = np.random.SeedSequence(ENTROPY + self.gen + self.seed*1000).generate_state(self.n_trials)
        # Todo: add multiprocessing.
        for i, agent in enumerate(self.population):
            for j in range(self.n_trials):
                fitness_scores[i] += collect_episode_reward(agent, seeds[j]) / self.n_trials
        return fitness_scores

    @torch.no_grad()
    def evolve_population(self, fitness_scores):
        new_population = []
        sorted_fitness_indices = np.argsort(-fitness_scores)

        if self.wandb:
            print()
            wandb.log({
                'topk/max_fitness_score': fitness_scores[sorted_fitness_indices[0]]
            }, step=self.gen)

        self.eval_agent(self.population[sorted_fitness_indices[0]])
        if self.keep_elites:
            for i in range(self.k):
                new_population.append(self.population[sorted_fitness_indices[i]])

        for i in range(self.population_size - len(new_population)):
            ancestor_index = sorted_fitness_indices[np.random.randint(self.k)]
            ancestor_agent = self.population[ancestor_index]
            new_agent = copy.deepcopy(ancestor_agent)
            for param in new_agent.parameters():
                new_param = param
                # Add random noise centered on current parameter values.
                noise_mask = torch.randn_like(new_param) < self.param_noise_prob
                new_param = new_param + self.param_noise_std * torch.randn_like(new_param) * noise_mask

                # Reset some parameters to 0 mean unit variance samples.
                mutation_mask = torch.rand_like(new_param)
                new_value = torch.where(mutation_mask < self.mutation_prob, torch.randn_like(new_param), new_param)
                param.copy_(new_value)
            new_population.append(new_agent)
        self.gen += 1
        return new_population


class EvolutionaryStrategies:

    def __init__(self):
        self.sigma = 0.5
        self.alpha = 0.05
        self.population_size = 100
        self.n_seeds = 3

    @torch.no_grad()
    def generate_modified_agents(self, agent):
        population = []
        for i in range(self.population_size):
            new_agent = copy.deepcopy(agent)
            epsilons = []
            for param in new_agent.parameters():
                epsilon = torch.randn_like(param)
                new_value = param + epsilon * self.sigma
                param.copy_(new_value)
                epsilons.append(epsilon)
            population.append((new_agent, epsilons))
        return population

    @torch.no_grad()
    def run(self, n_generations):
        agent = SimpleActor()
        for i in (pbar := trange(n_generations, desc='Generation')):
            seeds = np.random.SeedSequence(ENTROPY + i).generate_state(self.n_seeds)
            modified_population = self.generate_modified_agents(agent)
            gradient_estimates = [torch.zeros_like(param) for param in agent.parameters()]
            scale = self.alpha / (self.population_size * self.sigma)

            best_reward = -1E9
            # Accumulate the gradient estimate.
            for member in modified_population:
                modified_agent, epsilons = member
                avg_reward = 0
                for seed in seeds:
                    avg_reward += collect_episode_reward(modified_agent, seed) / len(seeds)
                best_reward = max(best_reward, avg_reward)
                for j, epsilon in enumerate(epsilons):
                    gradient_estimates[j] = gradient_estimates[j] + scale * avg_reward * epsilon

            pbar.set_postfix({'Best Reward':  best_reward})
            for j, param in enumerate(agent.parameters()):
                param.copy_(param + gradient_estimates[j])





if __name__ == "__main__":

    ea = SMCEvolutionaryAlgorithm(100)
    ea.train(1)

    # es = TopKEvolutionaryAlgorithm(100)
    # es.train(1000)
