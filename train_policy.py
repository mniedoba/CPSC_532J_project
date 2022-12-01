import hydra
import wandb

from evolutionary_algorithm import SMCEvolutionaryAlgorithm, TopKEvolutionaryAlgorithm

@hydra.main('conf', 'config.yaml', version_base=None)
def train(cfg):

    if cfg.wandb:
        wandb.init(entity='mniedoba', project='rl_smc')

    if cfg.algorithm == 'SMC':
        trainer = SMCEvolutionaryAlgorithm(cfg)
        trainer.train(cfg.max_generations)
    elif cfg.algorithm == 'TOPK':
        trainer = TopKEvolutionaryAlgorithm(cfg)
        trainer.train(cfg.max_generations)



if __name__ == '__main__':
    train()