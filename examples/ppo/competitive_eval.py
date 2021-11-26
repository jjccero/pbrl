import gym
import torch

from pbrl.algorithms.ppo import Policy
from pbrl.competitive import Agent, CompetitiveEnv


class DemoCompetitiveEnv(CompetitiveEnv):
    try:
        import gym_compete
    except ModuleNotFoundError:
        raise ModuleNotFoundError

    def init(self, env_name, config_policy):
        worker_ids = [4, 2]
        for i, worker_id in enumerate(worker_ids):
            filename_policy = 'result/{}-0-current/{}.pkl'.format(env_name, worker_id)
            agent = Agent(
                Policy(
                    observation_space=self.observation_space.spaces[i],
                    action_space=self.action_space.spaces[i],
                    **config_policy
                )
            )
            agent.load_from_dir(filename_policy)
            self.agents.append(agent)
            self.indices.append([i])

    def after_done(self):
        print(self.infos)


def main():
    config_policy = dict(
        rnn='lstm',
        hidden_sizes=[64, 64],
        activation=torch.nn.ReLU,
        obs_norm=True,
        critic=False,
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    )

    env_name = 'sumo-humans-v0'
    seed = 0
    torch.manual_seed(seed)

    env_test = DemoCompetitiveEnv(gym.make(env_name), config_policy=config_policy, env_name=env_name)
    env_test.seed(seed)

    env_test.reset()
    while True:
        _, _, done, info = env_test.step()
        env_test.render()
        if True in done:
            print(info)
            env_test.reset()


if __name__ == '__main__':
    main()
