import matplotlib.pyplot as plt

# Initialize empty lists to store data
episodes = []
agents = []
rewards = []
num_actions = []

# Read data from the log.log file
with open('training_log.log', 'r') as file:
    for line in file:
        if 'Episode' in line and 'Total Reward' in line:
            time_episode_str, agent_str, reward_str, num_actions_str, epsilon_str = line.split(', ')
            episode = int(time_episode_str.split(' ')[4])
            agent = agent_str.split(' ')[1]
            reward = float(reward_str.split(': ')[1])
            num_action = int(num_actions_str.split(' ')[2])
            episodes.append(episode)
            agents.append(agent)
            rewards.append(reward)
            num_actions.append(num_action)

window_size = 10

for agent in set(agents):
    agent_rewards = [r for i, r in enumerate(rewards) if agents[i] == agent]
    agent_episodes = [e for i, e in enumerate(episodes) if agents[i] == agent]
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    for i in range(len(agent_rewards) - window_size + 1):
        
        # Store elements from i to i+window_size
        # in list to get the current window
        window = agent_rewards[i : i + window_size]
    
        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)
        
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
        
        # Shift window to right by one position
        i += 1

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(agent_episodes[window_size-1:], moving_averages, marker='o', linestyle='-', label=agent)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards vs Episodes')
    plt.legend()
    plt.grid(True)

plt.show()
