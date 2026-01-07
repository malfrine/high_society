The goal of this project is to build an RL agent that plays the card game high society 

The rules of the game are here: https://www.ospreypublishing.com/media/1fohbutt/hisoc_rulebook_for_web.pdf

The motivation behind this is to get a deeper understanding of RL. I have been taking CS285 and I'm done most of the preliminary material. 
The homework assignments have been helpful but they have been lmiiting because there is a lot of boilerplate code. 

In this project, my focus is the RL algorithms. I will be using LLMs to help with the plumbing like analysis scripts, building the environment, etc
but the fundmantal goal is to learn and improve my understanding of RL algos. 


Brainstorming:

High society is a multi-person turn-based card game. Unlike the environments I've been using so far in CS285. 
I need to figure out how this is supported and how environments are built to support this. I'm sure there is some notion of self-play here.


What should the structure of the neural network be? 
It should be able to understand the state of the game and then it should tell us what it plans to do in this betting round?
And then how do I encode rules so it doesn't make illegal moves?
For example:
    - it should know that it can't bet 7k francs if it's already spent it.
    - it should know that it must always bet more than the next person


What is a good learning architecture? I'm making discrete systems and the "model" of the owrld is not clear to me. 
I think I should read up on alphago to get a sense of how they implemented it. I remember they used Q-learning but not much more than that.


Tasks:

1. Have agents randomly making moves and successfully playing the game  
2. Figure out the current neural network structure? Does the learning appraoch affect the structure I choose?
3. Figure how the best learning algorithm for problems like this. 

Brainstorming 2:

Just finished up a good brainstorming session with claude code:

There's different states to model: internal during bidding, internal normally, and external to agent

For internal information normally:
    - What bills the agent has
    - What value and prestige cards an agent has
    - Current value

For information during a bid:
    - The current bid size
    - The current bills that an agent has bid
    - The bills the other agents have bid?

For external information in general:
    - How many prestige cards have dropped
    - Which cards have dropped in the past?


Obviously the gym environment will track more state but I'm mostly just interested in specing out what the model sees. 


Internal info:
- agent bills: [0, 1, 0, ..., 1] # 1 if bill x is used else 0
- prestige cards: [0, 1, 0, ...] # 1 if agent has prestige card else 0
- value cards: [0, 0, ..., ] # 1 if agent has value card else 0

Bid info:
- internal: agent bills - drop bid ones too but same as above
- shared: current bid size, bills bid by other agents

during the bid the agent has a policy network that decides how much more to bid. 
then a separate network determines which cards to pick to match the bid. for now we can use a simple search algo to find a valid combination.

Opponent info:
- their current value
- do we want to encode the cards they have as well? i think we start off with this and pare it back as needed.


External env info:
- which prestige/value cards are remaining

for MVP
- agents are given full memory of other agent cards
- agents bill selection is deterministic
- we will not use the disgrace cards


Tasks 2:

1. validate this setup - are there things i am missing?
    - we should give the actual card number. 
2. determine the input and output of the neural networks and other things like layers etc. 
3. start off with a vanilla PG approach and we can complicate based on what we see



Brainstorming env

observation space: 
[
    current_winnings_amount: 0...100
    current_pot_size: 0 ... 100
    value_of_card_to_agent: -100 ... 100
    num_prestige_cards_remaining: 4...1
    total_money_remaining: 0...100
    player1_bid: 0 ... 100
    player2_bid: 0 ... 100
    ..
    player5_bid: 0 ... 100
]


action space:
[
    money_bet: 0...100
]


MVP:
- No denominations tracking
- No /2 or -5 cards yet


