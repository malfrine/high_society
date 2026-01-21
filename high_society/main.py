import numpy as np

from high_society.environment import SimpleHighSocietyEnv
from high_society.agents import VanillaPGAgent, RandomAgent, Agent
from high_society.utils import cat_dict_array, collect_trajectories



def main():
    # run 1000 games with random agents
    env = SimpleHighSocietyEnv(3)
    max_steps = 1000
    batch_size = 10_000
    training_steps = 1000
    agents: list[Agent] = [
        RandomAgent(player_id=0, obs_space=env.observation_space("player_0"), pass_probability=0.3),
        RandomAgent(player_id=1, obs_space=env.observation_space("player_1"), pass_probability=0.5),
        VanillaPGAgent(player_id=2, obs_space=env.observation_space("player_2"))
    ]
    for step in range(training_steps):
        batch_traj_data: list[dict[str, np.ndarray]] = []
        wins = 0
        for _ in range(batch_size):
            env.reset()
            traj_data = collect_trajectories(env, agents, max_steps=max_steps)
            traj = traj_data[2]  # Get player 2's trajectory
            batch_traj_data.append(traj)
            if traj["won"]:
                wins += 1
        pg_agent: VanillaPGAgent = agents[2]
        pg_agent.update(batch_traj_data)
        print(f"Step {step + 1}/{training_steps}: PG agent won {wins}/{batch_size} games ({100 * wins / batch_size:.1f}%)")

if __name__ == "__main__":
    main()