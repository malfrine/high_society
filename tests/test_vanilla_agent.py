from high_society.environment import SimpleHighSocietyEnv
from high_society.agents import VanillaPGAgent, RandomAgent
from high_society.utils import cat_dict_array

def test_vanilla_pg_agent_action():
    env = SimpleHighSocietyEnv(num_players=3)
    env.reset(seed=42)
    obs_space = env.observation_space("player_0")
    agent = VanillaPGAgent(player_id=0, obs_space=obs_space)
    obs_dict = env.observe("player_0")
    obs = cat_dict_array(obs_dict)
    action, log_prob = agent.get_action(obs)
    assert action.shape == (1,)
    assert isinstance(log_prob, float)
    bid = float(action[0])
    assert bid >= 0
    assert bid <= float(obs_dict["remaining_money"][0])

def test_random_agent_action():
    env = SimpleHighSocietyEnv(num_players=3)
    env.reset(seed=42)
    obs_space = env.observation_space("player_0")
    agent = RandomAgent(player_id=0, obs_space=obs_space, pass_probability=0.5, seed=123)
    obs_dict = env.observe("player_0")
    obs = cat_dict_array(obs_dict)
    action, log_prob = agent.get_action(obs)
    assert action.shape == (1,)
    assert isinstance(log_prob, float)
    bid = float(action[0])
    assert bid >= 0
    assert bid <= float(obs_dict["remaining_money"][0])