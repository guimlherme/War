# War game (backend)

This is an semi-complete version of the 'War' game (brazilian version of Risk).

A Deep Q-Learning agent was also implemented to train in this environment

## How to play

```bash
python -m game.war
```

## Rules

<https://pt.scribd.com/doc/41739031/Manual-Original-Do-War>

## How to train the agent

Default behaviour is to start traning with random agent and then following the option of last checkpoint (-r or -s)

```bash
python -m agent.model
```

If you want to train it with random agents:

```bash
python -m agent.model -r num_agents
```

Or if you want to train it agaisn't itself (selftrain) and extra random agents:

```bash
python -m agent.model -s num_agents
# only the first two agents will be DQN agents
```

## How the agent works

The agent receives a state vector that is composed of the current phase (reinforcement, attack, transfer), the agent's primary objective, and the pair (Owner, Number of Troops) for every territory in the game.

On each step, the agent decides to skip or act toward a territory. The valid territories are:

- Its own territories, in the reinforcement phase
- Enemy's territories, if they can be attacked, in the attack phase
- Its own territories, if they can receive troops, in the transfer phase

## Limitations

In case of attack, the neighboring territory with the most troops will attack. In case of transfer, the neighboring territory with the **least** troops will cede the troops. This was made to help concentrating the troops in the borders.

Unlike the original game, I didn't put any restrictions on the transfer phase.

To choose the number of troops in each action, the AI will always choose to use 20% of the available pool, rounded up. This can be changed in game/players/ai_player.py

To handle the cards, the AI will always trade ASAP.

The only identifies 3 entities: itself, enemies and target (for objectives). This means that all enemies that are not targets use the same identifier.
