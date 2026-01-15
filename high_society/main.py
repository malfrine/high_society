import numpy as np

from high_society.environment import SimpleHighSocietyEnv
from high_society.agents import VanillaPGAgent, RandomAgent, Agent
from high_society.utils import cat_dict_array, collect_trajectories



def main():
    # run 1000 games with random agents
    env = SimpleHighSocietyEnv(3)
    max_steps = 1000
    agents: list[Agent] = [
        RandomAgent(player_id=0, pass_probability=0.3),
        RandomAgent(player_id=1, pass_probability=0.5),
        VanillaPGAgent(player_id=2, obs_dim=env.obs_dim("player_2"))
    ]
    for _ in range(1000):
        env.reset()
        episode_data = collect_trajectories(env, agents)

        

if __name__ == "__main__":
    main()