from high_society.environment import SimpleHighSocietyEnv
from high_society.agents import Agent, VanillaPGAgent

import numpy as np


def cat_dict_array(d: dict[str, np.array]) -> np.array:
    return np.concatenate(
        [d[k] for k in sorted(d.keys())],
        axis=0
    )


def collect_trajectories(
    env: SimpleHighSocietyEnv,
    agents: list[Agent],
    max_steps: int = 1000
) -> dict[int, dict[str, np.ndarray]]:
    """Run a game episode and collect trajectory data for trainable agents.

    Args:
        env: The High Society environment
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

    # Convert lists to numpy arrays
    result: dict[int, dict[str, np.ndarray]] = {}
    for player_id, data in episode_data.items():
        result[player_id] = {
            "observations": np.array(data["observations"]),
            "actions": np.array(data["actions"]),
            "log_probs": np.array(data["log_probs"]),
            "rewards": np.array(data["rewards"]),
            "terminateds": np.array(data["terminateds"]),
            "truncateds": np.array(data["truncateds"]),
        }

    return result