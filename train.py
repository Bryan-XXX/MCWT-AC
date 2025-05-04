import argparse
import gym
import numpy as np
import time
import os
import wandb
from actor_network import Actor
from critic_network import Critic
from env_train import MyEnv
import random

OUTPUT_GRAPH = False
MAX_EPISODE = 1001
RENDER = False
LAMBDA = 0.9
LR_A = 0.0001
LR_C = 0.0001


def allFinish(dones):
    for i in dones:
        if i == 0:
            return False
    return True


def train(agent_num, env, actors, critics):
    s, dones, all_r = [], [0] * agent_num, [0] * agent_num
    average_rewards = 0
    data_dir = ''
    data_dir = os.path.join(data_dir, '')

    for user in range(agent_num):
        env.states[user] = [-1] * env.state_num

    for j in range(env.edges_num):
        env.edge_capacity[j] = random.uniform(3000, 4000)

    env.cloud_capacity = random.uniform(6000, 7000)

    env.velocity[0] = random.uniform(5500, 6500)
    env.velocity[1] = random.uniform(7000, 8000)
    env.velocity[2] = random.uniform(500, 1000)
    env.velocity[3] = random.uniform(1000, 1500)

    for user in range(env.agent_num):
        currStraight = env.straightMec[user] - 1
        env.states[user][1] = env.edge_capacity[currStraight]
        env.states[user][2] = env.cloud_capacity
        env.states[user][3:7] = env.velocity

    for i_agent_num in range(agent_num):
        s.append(env.reset(i_agent_num))

    for i_episode in range(MAX_EPISODE):
        while 1:
            for i_agent_num in range(agent_num):
                actor = actors[i_agent_num]
                critic = critics[i_agent_num]
                straightMec = env.straightMec[i_agent_num] - 1
                env.states[i_agent_num][9] = env.judgeZero(env.edgeTime[straightMec])
                s[i_agent_num] = env.states[i_agent_num]

                a = actor.choose_action(s[i_agent_num], env.curr_task[i_agent_num], env.localTasks[i_agent_num])
                s_new, r, done, _, _ = env.step(a, i_agent_num)
                s_new = np.array(s_new).astype(np.float32)

                all_r[i_agent_num] += r

                td_error = critic.learn(s[i_agent_num], r, s_new)

                actor.learn(s[i_agent_num], a, td_error)
                s[i_agent_num] = s_new

                if done:
                    ep_rs_sum = all_r[i_agent_num] / env.node_nums[i_agent_num]
                    average_rewards += ep_rs_sum
                    dones[i_agent_num] = 1
                    s[i_agent_num] = env.reset(i_agent_num)
                    all_r[i_agent_num] = 0

            if allFinish(dones):
                dones = [0] * agent_num
                print('Episode: {}/{}  | Episode Reward: {:.4f}' \
                      .format(i_episode, MAX_EPISODE, average_rewards / agent_num))
                average_rewards = 0

                for j in range(env.edges_num):
                    env.edge_capacity[j] = random.uniform(3000, 4000)

                env.cloud_capacity = random.uniform(6000, 7000)

                env.velocity[0] = random.uniform(5500, 6500)
                env.velocity[1] = random.uniform(7000, 8000)
                env.velocity[2] = random.uniform(500, 1000)
                env.velocity[3] = random.uniform(1000, 1500)

                for user in range(env.agent_num):
                    currStraight = env.straightMec[user] - 1
                    env.states[user][1] = env.edge_capacity[currStraight]
                    env.states[user][2] = env.cloud_capacity
                    env.states[user][3:7] = env.velocity

                break

        # if i_episode % 50 == 0 and i_episode != 0:
        #     for i_agent_num in range(agent_num):
        #         actor_save_path = os.path.join(data_dir,
        #                                        'model_actor_' + str(i_agent_num) + '_' + str(i_episode) + '.npz')
        #         actors[i_agent_num].save_ckpt(actor_save_path)


if __name__ == '__main__':

    agent_num = 20

    Actors, Critics = [], []
    env = MyEnv()
    N_F = env.state_num
    N_A = env.action_space.n

    for i in range(agent_num):
        actor = Actor(n_features=N_F, n_actions=N_A, name="actor" + str(i), lr=LR_A)
        critic = Critic(n_features=N_F, name="critic" + str(i), lr=LR_C)
        Actors.append(actor)
        Critics.append(critic)

    train(agent_num, env, Actors, Critics)
