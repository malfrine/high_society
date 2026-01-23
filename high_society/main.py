import numpy as np

from high_society.environments.simple import SimpleHighSocietyEnv
from high_society.environments.discrete import DiscreteHighSocietyEnv
from high_society.agents import VanillaPGAgent, RandomAgent, Agent, DiscreteAgent, DiscreteRandomAgent
from high_society.utils import cat_dict_array


def collect_trajectories_simple(
    env: SimpleHighSocietyEnv,
    agents: list[Agent],
    max_steps: int = 1000
) -> dict[int, dict[str, np.ndarray]]:
    """Run a game episode and collect trajectory data for trainable agents (continuous action space).

    Args:
        env: The SimpleHighSocietyEnv environment
        agents: List of agents (data only collected for VanillaPGAgent instances)
        max_steps: Maximum steps before truncating

    Returns:
        Dict mapping player_id to trajectory data containing:
        - observations: (T, obs_dim) array
        - actions: (T, action_dim) array
        - log_probs: (T,) array
        - rewards: (T,) array
        - terminateds: (T,) array
        - truncateds: (T,) array
    """
    env.reset()

    # Build lookup from agent name to agent object
    agent_lookup = {f"player_{agent.player_id}": agent for agent in agents}

    # Initialize data collection for trainable agents only
    episode_data: dict[int, dict[str, list]] = {}
    for agent in agents:
        if isinstance(agent, VanillaPGAgent):
            episode_data[agent.player_id] = {
                "observations": [],
                "actions": [],
                "log_probs": [],
                "rewards": [],
                "terminateds": [],
                "truncateds": [],
            }

    step_count = 0
    while not all(env.terminations.values()) and step_count < max_steps:
        agent_name = env.agent_selection
        agent = agent_lookup[agent_name]

        obs_dict = env.observe(agent_name)
        obs = cat_dict_array(obs_dict)
        action, log_prob = agent.get_action(obs)

        # Collect data for trainable agents
        if agent.player_id in episode_data:
            episode_data[agent.player_id]["observations"].append(obs)
            episode_data[agent.player_id]["actions"].append(action)
            episode_data[agent.player_id]["log_probs"].append(log_prob)
            episode_data[agent.player_id]["rewards"].append(env.rewards[agent_name])
            episode_data[agent.player_id]["terminateds"].append(env.terminations[agent_name])
            episode_data[agent.player_id]["truncateds"].append(env.truncations[agent_name])

        env.step(action)
        step_count += 1

    # Check if we hit max steps (truncation)
    truncated = step_count >= max_steps

    # Add final step data with terminal rewards
    for agent in agents:
        if agent.player_id in episode_data:
            agent_name = f"player_{agent.player_id}"
            # Update last step's terminated/truncated flags
            if episode_data[agent.player_id]["terminateds"]:
                episode_data[agent.player_id]["terminateds"][-1] = env.terminations[agent_name]
                episode_data[agent.player_id]["truncateds"][-1] = truncated
                # Update last reward with final reward
                episode_data[agent.player_id]["rewards"][-1] = env.rewards[agent_name]

    # Determine winner (highest final reward among all players)
    final_rewards = {name: env.rewards[name] for name in env.agents}
    max_reward = max(final_rewards.values())

    # Convert lists to numpy arrays
    result: dict[int, dict[str, np.ndarray]] = {}
    for player_id, data in episode_data.items():
        agent_name = f"player_{player_id}"
        won = final_rewards[agent_name] == max_reward and max_reward > 0
        # Propagate final reward to all steps (episode return)
        n_steps = len(data["observations"])
        episode_return = final_rewards[agent_name]
        result[player_id] = {
            "observations": np.array(data["observations"]),
            "actions": np.array(data["actions"]),
            "log_probs": np.array(data["log_probs"]),
            "rewards": np.full(n_steps, episode_return, dtype=np.float32),
            "terminateds": np.array(data["terminateds"]),
            "truncateds": np.array(data["truncateds"]),
            "won": won,
        }

    return result


# For backwards compatibility
collect_trajectories = collect_trajectories_simple


def collect_trajectories_discrete(
    env: DiscreteHighSocietyEnv,
    agents: list[DiscreteAgent],
    trainable_ids: set[int],
    max_steps: int = 1000
) -> dict[int, dict[str, np.ndarray]]:
    """Run a game episode and collect trajectory data for discrete action space.

    Args:
        env: The DiscreteHighSocietyEnv environment
        agents: List of DiscreteAgent instances
        trainable_ids: Set of player_ids to collect data for
        max_steps: Maximum steps before truncating

    Returns:
        Dict mapping player_id to trajectory data containing:
        - observations: (T, obs_dim) array
        - actions: (T,) array of action indices
        - action_masks: (T, num_actions) array of valid action masks
        - log_probs: (T,) array
        - rewards: (T,) array
        - terminateds: (T,) array
        - truncateds: (T,) array
        - won: bool
    """
    env.reset()

    agent_lookup = {f"player_{agent.player_id}": agent for agent in agents}

    # Initialize data collection for trainable agents
    episode_data: dict[int, dict[str, list]] = {}
    for player_id in trainable_ids:
        episode_data[player_id] = {
            "observations": [],
            "actions": [],
            "action_masks": [],
            "log_probs": [],
            "rewards": [],
            "terminateds": [],
            "truncateds": [],
        }

    step_count = 0
    while not all(env.terminations.values()) and step_count < max_steps:
        agent_name = env.agent_selection
        agent = agent_lookup[agent_name]

        obs_dict = env.observe(agent_name)
        obs = cat_dict_array(obs_dict)
        action_mask = env.get_action_mask(agent_name)
        action, log_prob = agent.get_action(obs, action_mask)

        # Collect data for trainable agents
        if agent.player_id in episode_data:
            episode_data[agent.player_id]["observations"].append(obs)
            episode_data[agent.player_id]["actions"].append(action)
            episode_data[agent.player_id]["action_masks"].append(action_mask.copy())
            episode_data[agent.player_id]["log_probs"].append(log_prob)
            episode_data[agent.player_id]["rewards"].append(env.rewards[agent_name])
            episode_data[agent.player_id]["terminateds"].append(env.terminations[agent_name])
            episode_data[agent.player_id]["truncateds"].append(env.truncations[agent_name])

        env.step(action)
        step_count += 1

    truncated = step_count >= max_steps

    # Update final step data
    for player_id in trainable_ids:
        if episode_data[player_id]["terminateds"]:
            agent_name = f"player_{player_id}"
            episode_data[player_id]["terminateds"][-1] = env.terminations[agent_name]
            episode_data[player_id]["truncateds"][-1] = truncated
            episode_data[player_id]["rewards"][-1] = env.rewards[agent_name]

    # Determine winner
    final_rewards = {name: env.rewards[name] for name in env.agents}
    max_reward = max(final_rewards.values())

    # Convert to numpy arrays
    result: dict[int, dict[str, np.ndarray]] = {}
    for player_id, data in episode_data.items():
        agent_name = f"player_{player_id}"
        won = final_rewards[agent_name] == max_reward and max_reward > 0
        n_steps = len(data["observations"])
        episode_return = final_rewards[agent_name]
        result[player_id] = {
            "observations": np.array(data["observations"]),
            "actions": np.array(data["actions"]),
            "action_masks": np.array(data["action_masks"]),
            "log_probs": np.array(data["log_probs"]),
            "rewards": np.full(n_steps, episode_return, dtype=np.float32),
            "terminateds": np.array(data["terminateds"]),
            "truncateds": np.array(data["truncateds"]),
            "won": won,
        }

    return result


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