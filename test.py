import numpy as np
import time
import os
from actor_network import Actor
from critic_network import Critic
from env_test import MyEnv


LR_A = 0.0001  # learning rate for actor
LR_C = 0.0001  # learning rate for critic
MAX_EPISODE = 1


def allFinish(dones):
    for i in dones:
        if i == 0:
            return False
    return True


def test(agent_num, env, Actors, s):

    dones = [0] * agent_num
    all_r = [0] * agent_num
    times = [-1] * agent_num
    t = [1] * agent_num

    for i_episode in range(MAX_EPISODE):
        while 1:
            for i_agent_num in range(agent_num):

                if dones[i_agent_num] == 1:
                    continue

                if times[i_agent_num] == -1:
                    times[i_agent_num] = time.time()

                actor = Actors[i_agent_num]

                straightMec = env.straightMec[i_agent_num] - 1
                env.states[i_agent_num][9] = env.judgeZero(env.edgeTime[straightMec])
                s[i_agent_num] = env.states[i_agent_num]

                a = actor.choose_action(s[i_agent_num], t[i_agent_num], env.localTasks[i_agent_num])

                s_new, r, done, _, _ = env.step(a, i_agent_num)
                s_new = np.array(s_new).astype(np.float32)

                all_r[i_agent_num] += r
                # print(all_r[i_agent_num])

                s[i_agent_num] = s_new
                t[i_agent_num] += 1
                if done:
                    ep_rs_sum = all_r[i_agent_num] / env.node_nums[i_agent_num]
                    dones[i_agent_num] = 1
                    print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}' \
                          .format(i_episode, MAX_EPISODE, ep_rs_sum, time.time() - times[i_agent_num]))
            if allFinish(dones):
                break

    outputTime = []
    for i_user_num in range(agent_num):
        outputTime.append(env.endTimes[i_user_num][env.node_nums[i_user_num] - 1])

    averageTime = 0
    for t in outputTime:
        # print(t)
        averageTime += t
    averageTime /= len(outputTime)
    print(averageTime)
    return averageTime


if __name__ == '__main__':
    start = time.perf_counter()
    agent_num = 20
    data_dir = ''
    minTime = 10000
    re = []
    for i in range(1):
        Actors, s = [], []
        env = MyEnv()
        N_F = env.state_num
        N_A = env.action_space.n
        for i_agent_num in range(agent_num):
            actor = Actor(n_features=N_F, n_actions=N_A, name="actor" + str(i_agent_num+agent_num*i), lr=LR_A)

            actor_save_path = os.path.join(data_dir, 'model_actor_'+str(i_agent_num)+'_1000.npz')

            actor.load_ckpt(actor_save_path)
            s.append(env.reset(i_agent_num))

            Actors.append(actor)

        averageTime = test(agent_num, env, Actors, s)
        if averageTime not in re:
            re.append(averageTime)

    end = time.perf_counter()
    print('Execute time is: '+str(end-start))
    print(re)
