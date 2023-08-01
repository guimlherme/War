import matplotlib.pyplot as plt

# Initialize empty lists to store data
episodes = []
rewards = []

# Read data from the log.log file
with open('training_log.log', 'r') as file:
    for line in file:
        if 'Episode' in line and 'Total Reward' in line:
            time_episode_str, _, reward_str = line.split(', ')
            episode = int(time_episode_str.split(' ')[4])
            reward = float(reward_str.split(': ')[1])
            episodes.append(episode)
            rewards.append(reward)

window_size = 10
moving_averages = []

# Loop through the array to consider
# every window of size 3
for i in range(len(rewards) - window_size + 1):
    
    # Store elements from i to i+window_size
    # in list to get the current window
    window = rewards[i : i + window_size]
  
    # Calculate the average of current window
    window_average = round(sum(window) / window_size, 2)
      
    # Store the average of current
    # window in moving average list
    moving_averages.append(window_average)
      
    # Shift window to right by one position
    i += 1

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(episodes[window_size-1:], moving_averages, marker='o', linestyle='-')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Rewards vs Episodes')
plt.grid(True)
plt.show()