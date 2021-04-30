import gym

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from travian_env import building_dict, building_names, res_names, village_info_dict, \
actions_corr, requirement_dict, npc_deltas, levels, max_levels_dict, TravianEnv
from itertools import chain

mod = 'culture accum'
g = 100
boost = 1
ban = 0

env = TravianEnv(village_info_dict, building_dict, g, 5184000,
                      requirement_dict, max_levels_dict, npc_deltas,
                      actions_corr, res_names, mod, boost)
env.reset()

n_actions = env.village_n*778 - 1
state_dim = 128

import torch
import torch.nn as nn
import torch.nn.functional as F

network = nn.Sequential()

network.add_module('layer1', nn.Linear(state_dim, 32))
network.add_module('relu1', nn.ReLU())
network.add_module('layer2', nn.Linear(32, 32))
network.add_module('relu1', nn.ReLU())
network.add_module('layer2', nn.Linear(32, 32))
network.add_module('relu1', nn.ReLU())
network.add_module('layer2', nn.Linear(32, 32))
network.add_module('relu1', nn.ReLU())
network.add_module('layer2', nn.Linear(32, 32))
network.add_module('relu1', nn.ReLU())
network.add_module('layer3', nn.Linear(32, n_actions))


def get_action(state, epsilon=0, cheating=0, waiting=0):
    """
    sample actions with epsilon-greedy policy
    recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
    """
    should_explore3 = np.random.binomial(n=1, p=waiting)
    if should_explore3:
        return np.random.choice([45, 46])

    # should_explore2 = np.random.binomial(n=1, p=cheating)
    # if should_explore2:
    #  chosen_action = np.random.choice(range(0, 8))
    #  return int(chosen_action)

    if not env.is_available_and_rr(413)[0] and len(env.villages_are_available) != 2:
        actions_range = range(50)
    elif not env.is_available_and_rr(413)[0] and len(env.villages_are_available) == 2:
        actions_range = chain(range(49), range(778, 826))
    else:
        actions_range = range(n_actions)

    state = torch.tensor(state, dtype=torch.float32)
    q_values = network(state).detach().numpy()
    for act in actions_range:
        if not env.is_available_and_rr(act)[0]:
            q_values[act] = 0

    q_values_dict = {q_values[i]: i for i in range(q_values.shape[0])}
    pos = np.argmax(q_values[actions_range])
    val = q_values[actions_range][pos]

    greedy_action = q_values_dict[val]
    should_explore = np.random.binomial(n=1, p=epsilon)

    if should_explore:
        chosen_action = np.random.choice(np.where(q_values[actions_range] != 0)[0])
    else:
        chosen_action = greedy_action

    return int(chosen_action)


def compute_td_loss(states, actions, rewards, next_states, is_done, gamma=0.999, check_shapes=False):
    """ Compute td loss using torch operations only """
    states = torch.tensor(
        states, dtype=torch.float32)  # shape: [batch_size, state_size]
    actions = torch.tensor(actions, dtype=torch.long)  # shape: [batch_size]
    rewards = torch.tensor(rewards, dtype=torch.float32)  # shape: [batch_size]
    # shape: [batch_size, state_size]
    next_states = torch.tensor(next_states, dtype=torch.float32)
    is_done = torch.tensor(is_done, dtype=torch.uint8)  # shape: [batch_size]

    # get q-values for all actions in current states
    predicted_qvalues = network(states)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[
        range(states.shape[0]), actions
    ]

    # compute q-values for all actions in next states
    predicted_next_qvalues = network(next_states)

    # compute V*(next_states) using predicted next q-values
    next_state_values = torch.max(predicted_next_qvalues, dim=-1)[0]
    assert next_state_values.dtype == torch.float32

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    target_qvalues_for_actions = rewards + gamma * next_state_values

    # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    target_qvalues_for_actions = torch.where(
        is_done, rewards, target_qvalues_for_actions)

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions -
                       target_qvalues_for_actions.detach()) ** 2)

    return loss


opt = torch.optim.Adam(network.parameters(), lr=0.001)


def generate_session(env, epsilon=0, cheating=0, waiting=0, train=False):
    """play env with approximate q-learning agent and train it at the same time"""
    total_reward = 0
    s = env.reset()
    done = False
    logs = [np.array(s)]
    while not done:
        a = get_action(s, epsilon=epsilon, cheating=cheating, waiting=waiting)
        all_info = env.step(a)

        st0 = list(env.X['village0'].values())
        del st0[-4]

        st1 = list(env.X['village1'].values())
        del st1[-4]

        st0.extend(st1)

        wow = st0

        wow.append(list(env.res_growths[0].values()))

        wow.append(list(env.res_growths[1].values()))
        wow.append([env.granary_capacities[0]])
        wow.append([env.granary_capacities[1]])
        wow.append([env.storage_capacities[0]])
        wow.append([env.storage_capacities[1]])
        wow.append([env.boost[0]])
        wow.append([env.boost[1]])
        wow.append([env.gold])
        wow.append([env.current_time])
        to_print = np.array([item for sublist in wow for item in sublist])
        next_s = to_print
        r = all_info[1]
        done = all_info[2]
        if train:
            opt.zero_grad()
            compute_td_loss([s], [a], [r], [next_s], [done]).backward()
            opt.step()
        if any(s[i] != next_s[i] for i in range(len(list(s)))):
            logs.append(np.insert(s, 0, a))
        total_reward += r
        s = next_s
    return total_reward, s, logs

rew = 0
epsilon = 0.7
cheating = 0
waiting = 0
for epoch in range(20):
    session_rewards = generate_session(env, epsilon=epsilon, cheating=cheating,
                                       waiting = waiting, train=True)
    if waiting <= 0.15:
      waiting *=0.999
    else:
      waiting *= 0.94
    epsilon *= 0.95
    cheating *= 0.99
    waiting *= 0.92

    print(session_rewards[0])
    print('session: ', generate_session(env, epsilon=0, cheating=0,
                                       waiting = 0, train=False)[0])