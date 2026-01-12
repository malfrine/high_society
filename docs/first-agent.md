okay so we now have (hopefully) a functioning gym environment to test with.

the first stage of training will be quite simple. we want to have some deep rl agent that plays against random agents and consistently beats them.

one thing i still have to figure out is how to mask cases where the agent would bid less than the allowed amount. it might make sense to have 2 outputs [[0 or 1], [bid_intensity or 0]]

architecture of the v0 RL agent

-
