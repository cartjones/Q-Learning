import gymnasium as gym
import imageio
from IPython.display import Image, display
import random

"""
When originally writing this project I used Googled Colab.
Due to it's hardware limitations for the Taxi Implementation
I had to 'train' the Taxi agent before adding all of the 
frames to the .gif file or else the program would run out of
ram and crash. 

Depending on the hardware of your computer this may not be
necessary, but still this way makes the program run faster.
"""


# Frozen Lake Implementation
env = gym.make('FrozenLake-v1',render_mode="rgb_array")
observation, info = env.reset()

frames = [] # keep a list of all the rendered frames in the animation

# preparing for q learning equation
state = 0
alpha = 0.1
gamma = 0.6

Q_frozen = {}
possible_states_frozen = range(env.observation_space.n)
possible_actions_frozen = range(env.action_space.n)

for s in possible_states_frozen:
    Q_frozen[s] = {}
    for a in possible_actions_frozen:
        Q_frozen[s][a] = 0

num_iterations = 10000
for i in range(num_iterations):
    frames.append( env.render() ) # render the next frame, append to frames list

# exploitation implementation
    if random.random() < i/num_iterations:
      best_estimated_reward = float("-inf")
      action = None # initializing action

      for a in possible_actions_frozen:
          if Q_frozen[state][a] > best_estimated_reward:
              best_estimated_reward = Q_frozen[state][a]
              action = a
    else:
      action = random.choice(range(env.action_space.n))

    observation, reward, terminated, truncated, info = env.step(action)
    next_state = observation

# finding the best q val
    next_state_Q_val_list = Q_frozen[next_state].values()
    next_state_bestq = max(next_state_Q_val_list)

# q learning update rule
    Q_frozen[state][action] = Q_frozen[state][action] + alpha*(reward + gamma*(next_state_bestq) - Q_frozen[state][action])

    state = observation # saving the current state for the next loop iteration

    if terminated or truncated:
        observation, info = env.reset()

env.close()

# save the animation as animation.gif and then display it in the notebook
imageio.mimsave('frozen_lake.gif', frames, fps=30)  # fps: frames per second
display(Image(filename='frozen_lake.gif'))


# Taxi Implementation
env = gym.make('Taxi-v3', render_mode = "rgb_array")
observation, info = env.reset()

# creating Q table
Q_taxi = {}
possible_states_taxi = range(env.observation_space.n)
possible_actions_taxi = range(env.action_space.n)

for s in possible_states_taxi:
    Q_taxi[s] = {}
    for a in possible_actions_taxi:
        Q_taxi[s][a] = 0

count = 0
# preparing for Q learning equation
state = 0
alpha = 0.1
gamma = 0.6

num_iterations = 1000000 # "training" the agent without rendering each frame
for i in range(num_iterations):

# exploitation implementation
    if random.random() < i/num_iterations:
      best_estimated_reward = float("-inf")
      action = None # initializing action

      for a in possible_actions_taxi:
          if Q_taxi[state][a] > best_estimated_reward:
              best_estimated_reward = Q_taxi[state][a]
              action = a
    else:
      action = random.choice(range(env.action_space.n))

    observation, reward, terminated, truncated, info = env.step(action)
    next_state = observation

# finding the best q val
    next_state_Q_val_list = Q_taxi[next_state].values()
    next_state_bestq = max(next_state_Q_val_list)

# q learning update rule
    Q_taxi[state][action] = Q_taxi[state][action] + alpha*(reward + gamma*(next_state_bestq) - Q_taxi[state][action])

    state = observation # saving the current state for the next loop iteration

    if terminated or truncated:
        count += 1
        observation, info = env.reset()

env.close() 

# repeating process now adding frames to .gif file
frames = []
state = 0
alpha = 0.1
gamma = 0.6

num_iterations = 100
for i in range(num_iterations):
    frames.append( env.render() ) # render the next frame, append to frames list

# exploitation implementation
    if random.random() < i/num_iterations:
      best_estimated_reward = float("-inf")
      action = None # initializing action

      for a in possible_actions_taxi:
          if Q_taxi[state][a] > best_estimated_reward:
              best_estimated_reward = Q_taxi[state][a]
              action = a
    else:
      action = random.choice(range(env.action_space.n))

    observation, reward, terminated, truncated, info = env.step(action)
    next_state = observation

# finding the best q val
    next_state_Q_val_list = Q_taxi[next_state].values()
    next_state_bestq = max(next_state_Q_val_list)

# q learning update rule
    Q_taxi[state][action] = Q_taxi[state][action] + alpha*(reward + gamma*(next_state_bestq) - Q_taxi[state][action])

    state = observation # saving the current state for the next loop iteration

    if terminated or truncated:
        observation, info = env.reset()

env.close()

# save the animation as animation.gif and then display it in the notebook
imageio.mimsave('taxi.gif', frames, fps=30)  # fps: frames per second

display(Image(filename='taxi.gif'))