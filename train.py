# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
from copy import deepcopy
from collections import deque
import torch.autograd
import numpy as np

__all__ = ['training', 'testing', 'wrap_np', 'np', 'gen_rand_data']


def wrap_np(state, device):
    return torch.from_numpy(state[np.newaxis, :]).float().unsqueeze(0).to(device)


def gen_rand_data(opt):
    state = []
    for bc in opt.ZC.boundary:
        state.append(np.random.randint(bc[0], bc[1]))
    state = np.array(state)
    return state


# define take action function
def take_action(opt, state1, action, t):
    index = action // 2
    act = action % 2
    state2 = deepcopy(state1)
    if act == 0:
        state2[index] -= max(np.floor(5 * opt.STEP_DECAY ** t), 1)
        if state2[index] < opt.ZC.boundary[index][0]:
            state2[index] += 2*max(np.floor(5 * opt.STEP_DECAY ** t), 1)
    else:
        state2[index] += max(np.floor(5 * opt.STEP_DECAY ** t), 1)
        if state2[index] > opt.ZC.boundary[index][1]:
            state2[index] -= 2*max(np.floor(5 * opt.STEP_DECAY ** t), 1)
    reward_ = opt.ZC.zf(state1) - opt.ZC.zf(state2)
    if opt.ZC.zf(state2) <= opt.CRITERION:
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
        # initial state for x and y
        # state = np.random.randint(opt.LOW_BOND, opt.HIGH_BOND, opt.NUM_VARIABLE)
        state = gen_rand_data(opt)
        for t in range(opt.MOST_BEAR_STEP):
            action, log_prob = policy.act(wrap_np(state, device))
            saved_log_probs.append(log_prob)
            state, reward, done = take_action(opt, state, action, t)
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


def testing(opt, policy, device):
    # state = np.random.randint(opt.LOW_BOND, opt.HIGH_BOND, opt.NUM_VARIABLE)
    state = gen_rand_data(opt)

    init_state = deepcopy(state)
    action, _ = policy.act(wrap_np(state, device))
    state, reward, done = take_action(opt, state, action, 0)
    policy.eval()
    for step in range(500):
        action, _ = policy.act(wrap_np(state, device))
        state, reward, done = take_action(opt, state, action, step)
        if done:
            print("==> Starting from (%.2f, %.2f). Use %d steps to terminate(%.2f, %.2f), max=%.2f." %
                  (init_state[0], init_state[1], step, state[0], state[1], -opt.ZC.zf(state)/1000))
            break
    print("==> Testing finished.")
    return step
