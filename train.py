# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
from copy import deepcopy
from collections import deque
import torch.autograd
import numpy as np


def wrap_np(state, device):
    return torch.from_numpy(state[np.newaxis, :]).float().unsqueeze(0).to(device)


def zf(x, y):
    x = 0.1 * x
    y = 0.1 * y
    r = np.sqrt(x ** 2 + y ** 2)
    z = np.sin(x ** 2 + 3 * y ** 2) / (0.1 + r ** 2) + (x ** 2 + 5 * y ** 2) * np.exp(1 - r ** 2) / 2
    return -z * 1000


def zfzurab(x,y):
    pass


# define take action function
def take_action(state1, action, t, decay=0.8):
    # action 0: x plus 0.1; action 1: x minus 0.1; action 2: y plus 0.1; action 3: y minus 0.1
    state2 = deepcopy(state1)
    if action == 0:
        state2[0] = state1[0] + max(np.floor(5 * decay ** t), 1)
    elif action == 1:
        state2[0] = state1[0] - max(np.floor(5 * decay ** t), 1)
    elif action == 2:
        state2[1] = state1[1] + max(np.floor(5 * decay ** t), 1)
    else:
        state2[1] = state1[1] - max(np.floor(5 * decay ** t), 1)
    reward_ = zf(state1[0], state1[1]) - zf(state2[0], state2[1])
    if zf(state2[0], state2[1]) <= -3650:
        done = True
        print("==> Terminate. %d steps used." % t)
    else:
        done = False
    return state2, reward_, done


def training(opt, writer, policy, device, pre_epoch=0):
    scores_deque = deque(maxlen=opt.QUEUE_LENGTH)
    optimizer = torch.optim.Adam(policy.parameters(), lr=opt.LEARNING_RATE)
    for epoch in range(1, opt.NUM_EPOCHS + 1):
        saved_log_probs = []
        rewards = []
        state = np.ndarray([2])
        # initial state for x and y
        state[0] = np.random.randint(-20, 20)
        state[1] = np.random.randint(-20, 20)
        for t in range(opt.MOST_BEAR_STEP):
            action, log_prob = policy.act(wrap_np(state, device))
            saved_log_probs.append(log_prob)
            state, reward, done = take_action(state, action, t, decay=opt.STEP_DECAY)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))

        discounts = [opt.GAMMA ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if epoch % opt.PRINT_EVERY == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(epoch + pre_epoch, np.mean(scores_deque)))

        # Add summary to tensorboard
        writer.add_scalar("Scores", np.mean(scores_deque), epoch + pre_epoch)

        # Save the model
        if epoch % opt.PRINT_EVERY == 0:
            policy.mt_save(pre_epoch + epoch, np.mean(scores_deque))
    print('==> Training Finished.')
    return policy


def testing(policy, device):
    state = np.random.randint(-20, 20, 2)
    init_state = deepcopy(state)
    action, _ = policy.act(wrap_np(state, device))
    state, reward, done = take_action(state, action, 0)
    policy.eval()
    for step in range(2000):
        action, _ = policy.act(wrap_np(state, device))
        state, reward, done = take_action(state, action, step, decay=0.7)
        if done:
            print("==> Starting from (%.2f, %.2f). Use %d steps to terminate." % (init_state[0], init_state[1], step))
            break
    print("==> Testing finished.")
    return step
