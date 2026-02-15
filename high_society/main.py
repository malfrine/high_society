from collections import defaultdict
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from high_society.environments.simple import SimpleHighSocietyEnv
from high_society.environments.discrete import DiscreteHighSocietyEnv
from high_society.agents import VanillaPGAgent, RandomAgent, Agent, DiscreteAgent, DiscreteRandomPassAgent, DQNAgent
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


def collect_trajectories_discrete(
    env: DiscreteHighSocietyEnv,
    agents: list[DiscreteAgent],
    max_steps: int = 1000
) -> dict[int, dict[str, np.ndarray]]:
    """Run a game episode and collect trajectory data for discrete action space.

    Args:
        env: The DiscreteHighSocietyEnv environment
        agents: List of DiscreteAgent instances
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
    for agent in agents:
        episode_data[agent.player_id] = {
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
    for agent in agents:
        if agent.player_id in episode_data:
            agent_name = f"player_{agent.player_id}"
            episode_data[agent.player_id]["terminateds"][-1] = env.terminations[agent_name]
            episode_data[agent.player_id]["truncateds"][-1] = truncated
            episode_data[agent.player_id]["rewards"][-1] = env.rewards[agent_name]

    final_rewards = {name: env.rewards[name] for name in env.agents}
    max_reward = max(final_rewards.values())

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

def run_sessions(num_sessions: int, batch_size: int, training_steps: int, max_steps: int) -> DQNAgent:

    dqn_agent = None
    wins_by_pass_prob: dict[float, int] = defaultdict(int)
    games_by_pass_prob: dict[float, int] = defaultdict(int)

    writer = SummaryWriter("runs/dqn_high_society")

    for session in range(num_sessions):
        num_random_agents = np.random.randint(2, 4)
        env = DiscreteHighSocietyEnv(num_players=num_random_agents + 1)
        if dqn_agent is None:
            dqn_agent = DQNAgent(player_id=0, num_actions=env.num_actions, obs_space=env.observation_space("player_0"), epsilon=0.1)
            if os.path.exists("./experiments/results/dqn_agent.pth"):
                dqn_agent.q_net.load_state_dict(torch.load("./experiments/results/dqn_agent.pth"))
        random_agents: list[DiscreteAgent] = [
            DiscreteRandomPassAgent(player_id=i, pass_probability=np.random.uniform(0.0, 1.0), seed=43 + i)
            for i in range(1, num_random_agents + 1)
        ]
        agents: list[DiscreteAgent] = [dqn_agent] + random_agents

        for step in range(training_steps):
            batch_traj_data: list[dict[str, np.ndarray]] = []
            wins = 0
            for _ in range(batch_size):
                env.reset()
                traj_data = collect_trajectories_discrete(env, agents, max_steps=max_steps)
                traj = traj_data[0]  # Get DQN agent's trajectory (player 0)
                batch_traj_data.append(traj)
                if traj["won"]:
                    wins += 1

                for agent in random_agents:
                    pass_prob_bucket = round(agent.pass_probability * 10) / 10.0
                    if traj["won"]:
                        wins_by_pass_prob[pass_prob_bucket] += 1
                    games_by_pass_prob[pass_prob_bucket] += 1

            metrics = dqn_agent.update(batch_traj_data)

            global_step = session * training_steps + step
            win_rate = wins / batch_size
            cumulative_win_rate = sum(wins_by_pass_prob.values()) / sum(games_by_pass_prob.values())

            writer.add_scalar("train/win_rate", win_rate, global_step)
            writer.add_scalar("train/cumulative_win_rate", cumulative_win_rate, global_step)
            writer.add_scalar("train/loss", metrics["loss"], global_step)
            writer.add_scalar("q_values/mean_predicted", metrics["mean_q"], global_step)
            writer.add_scalar("q_values/mean_target", metrics["mean_target"], global_step)
            writer.add_scalar("q_values/error", metrics["q_error"], global_step)
            for bucket in sorted(wins_by_pass_prob.keys()):
                total_games = games_by_pass_prob[bucket]
                total_wins = wins_by_pass_prob[bucket]
                bucket_win_rate = total_wins / total_games if total_games > 0 else 0
                writer.add_scalar(f"win_rate_by_pass_prob/{bucket:.1f}", bucket_win_rate, global_step)
            writer.flush()

            print(f"Session {session + 1}/{num_sessions}: Step {step + 1}/{training_steps}: DQN agent won {wins}/{batch_size} games ({100 * win_rate:.1f}%)")
            print("\n=== Win Rate Metrics ===\n")
            for bucket in sorted(wins_by_pass_prob.keys()):
                total_games = games_by_pass_prob[bucket]
                total_wins = wins_by_pass_prob[bucket]
                bucket_win_rate = 100 * total_wins / total_games if total_games > 0 else 0
                print(f"Pass prob {bucket:.1f}: {total_wins}/{total_games} wins ({bucket_win_rate:.1f}%)")
            print(f"Cumulative win rate: {100 * cumulative_win_rate:.1f}%")

    writer.close()
    return dqn_agent


# def run_self_play(env: DiscreteHighSocietyEnv, learning_agent: DQNAgent, num_dqn_agents: int, num_random_agents: int, max_steps: int, training_steps: int, batch_size: int) -> None:
#     assert num_dqn_agents + num_random_agents + 1 <= 5

#     dqn_agents: list[DQNAgent] = [
#         DQNAgent(player_id=i, num_actions=env.num_actions, obs_space=env.observation_space(f"player_{i+1}"), epsilon=0.1)
#         for i in range(num_dqn_agents)
#     ]
#     random_agents: list[DiscreteAgent] = [
#         DiscreteRandomPassAgent(player_id=i, pass_probability=np.random.uniform(0.0, 1.0), seed=43 + i)
#         for i in range(num_dqn_agents, num_dqn_agents + num_random_agents)
#     ]
    
    
#     for i in range(num_sessions):
#         batch_traj_data: list[dict[str, np.ndarray]] = []
#         wins = 0
#         for _ in range(batch_size):
#             env.reset()
#             traj_data = collect_trajectories_discrete(env, agents, max_steps=max_steps)
#             traj = traj_data[0]  # Get DQN agent's trajectory (player 0)
#             batch_traj_data.append(traj)
#             if traj["won"]:
#                 wins += 1

#             for agent in random_agents:
#                 pass_prob_bucket = round(agent.pass_probability * 10) / 10.0
#                 if traj["won"]:

#             metrics = dqn_agent.update(batch_traj_data)

#     writer.close()

def main():
    max_steps = 500
    batch_size = 100
    training_steps = 10
    num_sessions = 4000
    dqn_agent = run_sessions(num_sessions, batch_size, training_steps, max_steps)
    torch.save(dqn_agent.q_net.state_dict(), "./experiments/results/dqn_agent.pth")
    
if __name__ == "__main__":
    main()