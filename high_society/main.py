from collections import defaultdict
import os
from datetime import datetime
import random

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
        assert agent_name in agent_lookup, f"{agent_lookup.keys()}, {agent_name}"
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

    now = datetime.now()
    writer = SummaryWriter(f"runs/random_sessiosn_{now}")

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

def run_self_play(env: DiscreteHighSocietyEnv, learning_agent: DQNAgent, dqn_pool: list[str], max_steps: int, training_steps: int, batch_size: int) -> None:

    now = datetime.now()
    writer = SummaryWriter(f"runs/self_play{now}")

    wins_by_agent_class: dict[str, int] = defaultdict(int)
    games_by_agent_class: dict[str, int] = defaultdict(int)
    total_games = 0
    total_wins = 0
    
    for step in range(training_steps):

        num_opponents = random.randint(2, 4)

        env.reset(num_players=num_opponents + 1)
        # pick dqn agents with 75% probability
        num_dqn_agents = np.random.binomial(num_opponents, 0.5)
        num_random_agents = num_opponents - num_dqn_agents

        batch_traj_data: list[dict[str, np.ndarray]] = []
        wins_in_batch = 0

        dqn_agents: list[DQNAgent] = [
            DQNAgent(player_id=i + 1, num_actions=env.num_actions, obs_space=env.observation_space(f"player_{i+1}"), epsilon=0.1)
            for i in range(num_dqn_agents)
        ]
        dqn_agent_paths = []
        for a in dqn_agents:
            selected_dqn = dqn_pool[random.randint(0, len(dqn_pool) - 1)]
            dqn_agent_paths.append(selected_dqn)
            a.q_net.load_state_dict(torch.load(f"./experiments/results/pool/{selected_dqn}"))

        random_agents: list[DiscreteAgent] = [
            DiscreteRandomPassAgent(player_id=i + num_dqn_agents + 1, pass_probability=np.random.uniform(0.0, 1.0), seed=43 + i)
            for i in range(num_random_agents)
        ]

        agents = [learning_agent]
        agents.extend(dqn_agents)
        agents.extend(random_agents)

        for _ in range(batch_size):
            traj_data = collect_trajectories_discrete(env, agents, max_steps=max_steps)
            learning_agent_traj = traj_data[0]  # Get learning agent's trajectory (player 0)
            batch_traj_data.extend(traj_data.values())
            won = learning_agent_traj["won"]
            if won:
                wins_in_batch += 1

            games_by_agent_class["learning_agent"] += 1
            if won:
                wins_by_agent_class["learning_agent"] += 1
            for i, agent in enumerate(dqn_agents):
                agent_class = dqn_agent_paths[i]
                games_by_agent_class[agent_class] += 1
                if won:
                    wins_by_agent_class[agent_class] += 1
            for agent in random_agents:
                agent_class = f"random_{round(agent.pass_probability * 10) / 10.0:.1f}"
                games_by_agent_class[agent_class] += 1
                if won:
                    wins_by_agent_class[agent_class] += 1
        metrics = learning_agent.update(batch_traj_data)
        total_games += batch_size
        total_wins += wins_in_batch

        batch_win_rate = wins_in_batch / batch_size
        cumulative_win_rate = total_wins / total_games


        writer.add_scalar("train/win_rate", batch_win_rate, step)
        writer.add_scalar("train/cumulative_win_rate", cumulative_win_rate, step)
        writer.add_scalar("train/loss", metrics["loss"], step)
        writer.add_scalar("q_values/mean_predicted", metrics["mean_q"], step)
        writer.add_scalar("q_values/mean_target", metrics["mean_target"], step)
        writer.add_scalar("q_values/error", metrics["q_error"], step)

        print(f"Step {step + 1}/{training_steps}: Win rate: {100 * batch_win_rate:.1f}%")
        print(f"Cumulative win rate: {100 * cumulative_win_rate:.1f}%")
        for agent_class in wins_by_agent_class.keys():
            class_games = games_by_agent_class[agent_class]
            class_wins = wins_by_agent_class[agent_class]
            bucket_win_rate = 100 * class_wins / class_games if class_games > 0 else 0
            writer.add_scalar(f"win_rate_by_opponent/{agent_class}", bucket_win_rate / 100, step)
            print(f"\t{agent_class}: {class_wins}/{class_games} wins ({bucket_win_rate:.1f}%)")

        writer.flush()

    writer.close()

    return learning_agent

def run_tournament(max_steps: int, training_steps: int, batch_size: int, sessions: int) -> None:
    learning_agent_path = "./experiments/results/pool/dqn_agent_v3.pth"

    version = 4
    for session in range(sessions):
        print(f"----------STARTING SESSION {session}-----------")
        dqn_pool = [f for f in os.listdir("./experiments/results/pool") if ".pth" in f]
        env = DiscreteHighSocietyEnv()
        learning_agent = DQNAgent(player_id=0, num_actions=env.num_actions, obs_space=env.observation_space("player_0"), epsilon=0.1)
        if not os.path.exists(learning_agent_path):
            raise FileNotFoundError(f"{learning_agent_path} does not exist.")
        learning_agent.q_net.load_state_dict(torch.load(learning_agent_path))
        learning_agent = run_self_play(env, learning_agent, dqn_pool, max_steps, training_steps, batch_size)
        torch.save(learning_agent.q_net.state_dict(), f"./experiments/results/pool/dqn_agent_v{version}.pth")
        version += 1
        print(f"----------FINISHED SESSION {session}-----------")

        

def main():
    max_steps = 500
    batch_size = 100
    training_steps = 100_000
    num_sessions = 10
    run_tournament(max_steps, training_steps, batch_size, num_sessions)
    

    
    
if __name__ == "__main__":
    main()