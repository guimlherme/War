# War game (backend)

This is an semi-complete version of the 'War' game (brazilian version of Risk).

A Deep Q-Learning agent was also implemented to train in this environment

## How to play:

```bash
python -m game.war
```

## Rules:

https://pt.scribd.com/doc/41739031/Manual-Original-Do-War

## How to train the agent:

Default behaviour is to start traning with random agent and then following the option of last checkpoint (-r or -s)

```bash
python -m agent.model
```

If you want to train it with a random agent:

```bash
python -m agent.model -r
```

Or if you want to train it agaisn't itself (selftrain):

```bash
python -m agent.model -s
```
