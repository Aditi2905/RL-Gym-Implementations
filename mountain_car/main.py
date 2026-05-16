import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
import random
import itertools
import math
from time import sleep
from yaspin import yaspin

from utils.replay_memory import ReplayMemory, Transition

# ----- PARAMETERS ------ #
N_OBSERVATIONS = 2
N_ACTIONS = 3
# --- HYPERPARAMETERS --- #
BATCH_SIZE = 128
DQN_HIDDEN_LAYER_1_SIZE = 128
DQN_HIDDEN_LAYER_2_SIZE = 128
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
GAMMA = 0.99
LR = 3e-4
REPLAY_MEMORY_CAPACITY = 10000
TRAINING_EPISODE_COUNT = 300
TAU = 0.005
# ----------------------- #

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, DQN_HIDDEN_LAYER_1_SIZE)
        self.layer2 = nn.Linear(DQN_HIDDEN_LAYER_1_SIZE, DQN_HIDDEN_LAYER_2_SIZE)
        self.layer3 = nn.Linear(DQN_HIDDEN_LAYER_2_SIZE, n_actions)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
memory = ReplayMemory(REPLAY_MEMORY_CAPACITY)
policy_net = DQN(N_OBSERVATIONS, N_ACTIONS)
target_net = DQN(N_OBSERVATIONS, N_ACTIONS)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    action_batch = torch.cat(batch.action)
    state_batch = torch.cat(batch.state)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

env = gym.make("MountainCar-v0")
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        # We use our knowledge
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # We explore
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long)

spinner = yaspin(text=f"Model training... 0/{TRAINING_EPISODE_COUNT}")
spinner.start()

for i_episode in range(TRAINING_EPISODE_COUNT):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    for t in itertools.count():
        action = select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        reward = torch.tensor([reward])

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = TAU*policy_net_state_dict[key] + (1-TAU)*target_net_state_dict[key]
        target_net.load_state_dict(target_net_state_dict)

        if done:
            break

    spinner.text = f"Model training... {i_episode}/{TRAINING_EPISODE_COUNT}"

spinner.stop()

env.close()
print("Done with training... Let us test!")
for _ in range(10):
    demo_env = gym.make("MountainCar-v0", render_mode="human")
    state, _ = demo_env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    for _ in itertools.count():
        action = target_net(state).max(1).indices.view(1, 1)
        next_state, reward, terminated, truncated, _ = demo_env.step(action.item())
        demo_env.render()
        sleep(0.1)
        if terminated or truncated:
            break
        state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
    demo_env.close()
