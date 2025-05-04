import gym
import numpy as np
import pandas as pd
from gym.spaces import Discrete, Box
from numpy import random

from code_offloading import getReward
from graph import Graph


class MyEnv(gym.Env):
    def __init__(self):
        self.viewer = None
        self.agent_num = 20
        self.edges_num = 3
        self.max_task_num = [5, 6, 7]
        self.mecs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.straightMec = [0] * self.agent_num
        self.tasks = [[-1 for i in range(20)] for j in
                      range(self.agent_num)]
        self.state_num = len(self.mecs) + len(self.tasks[0])

        self.fix_node_num = 35

        self.states = [[] for i in range(self.agent_num)]
        self.node_nums = [0 for i in range(self.agent_num)]
        self.localTasks = [[] for i in range(self.agent_num)]
        self.taskTypes = [[-1 for i in range(self.fix_node_num)] for j in range(self.agent_num)]
        self.preEndTimes = [0 for i in range(self.agent_num)]
        self.endTimes = [[] for i in range(self.agent_num)]
        self.maxEnds = [[] for i in range(self.agent_num)]
        self.curr_task = [0 for i in range(self.agent_num)]
        self.graphs = [[] for i in range(self.agent_num)]
        self.edgeRelations = [[] for i in range(self.agent_num)]
        self.pres = [[] for i in range(self.agent_num)]
        self.weights = [[] for i in range(self.agent_num)]

        self.local_capacity = [2208, 2238, 1952, 2478, 2243, 1924, 1584, 2472, 1673, 1967, 2087, 1748, 1840, 2094, 1959,
                               1795, 2343, 2008, 1804, 1615, 1758, 1541, 2005, 1850, 2116, 2171, 1881, 1703, 1875, 2326,
                               2208, 2250, 2149, 2002, 2081, 2176, 1607, 1732, 2420, 2076, 2176, 2085, 1715, 1955, 1805,
                               2125, 2302, 1944, 2392, 1896]

        self.edge_capacity = [3051.128292750158, 3412.7797342310387, 3267.5573]
        self.cloud_capacity = 6054
        self.velocity = [6927.049428751091, 7682.767442696975, 1632.928192225152, 2301.6044313719417]

        for i_agent_num in range(self.agent_num):
            if i_agent_num % self.edges_num == 0:
                self.straightMec[i_agent_num] = 1
            if i_agent_num % self.edges_num == 1:
                self.straightMec[i_agent_num] = 2
            if i_agent_num % self.edges_num == 2:
                self.straightMec[i_agent_num] = 3
        self.mecs[0] = self.local_capacity[0]
        self.mecs[1] = self.edge_capacity[0]
        self.mecs[2] = self.cloud_capacity
        self.mecs[3:7] = self.velocity
        self.mecs[7] = 1
        self.mecs[8] = self.max_task_num[0]
        self.mecs[9] = 0

        self.localTime, self.edgeTime = [0] * self.agent_num, [[0] for i in range(
            self.edges_num)]

        self.action_space = Discrete(5)

    def get_reward3(self, taskNo, userNo, allTime):
        minTime = min(allTime)
        maxTime = max(allTime)
        currTime = allTime[self.taskTypes[userNo][taskNo] - 1]
        # print(self.taskTypes[userNo][taskNo], allTime[self.taskTypes[userNo][taskNo] - 1])
        # print(minTime, maxTime, currTime)
        # print(-(currTime - minTime)/(maxTime-minTime) )
        # return -(currTime - minTime)/(maxTime-minTime)
        # return minTime - currTime
        return maxTime - currTime

    def get_reward2(self, taskNo, userNo):
        currentVert = self.graphs[userNo].getVertex(taskNo)
        parents = self.graphs[userNo].getParents(currentVert)
        time = 0
        for parent in parents:
            time = max(time, self.endTimes[userNo][parent])
        return time - self.endTimes[userNo][taskNo]

    def get_env_back(self, userNo):
        taskNo = self.curr_task[userNo]
        re = getReward(taskNo, self.states[userNo][len(self.mecs):len(self.mecs) + 20],
                       self.taskTypes[userNo], self.endTimes[userNo],
                       self.localTime[userNo], self.edgeTime,
                       self.local_capacity[userNo], self.edge_capacity, self.cloud_capacity, self.velocity,
                       self.max_task_num, self.states[userNo][7])

        self.localTime[userNo] = re[0]
        self.edgeTime = re[1]
        self.endTimes[userNo][taskNo] = re[2]
        self.curr_task[userNo] += 1

        if self.taskTypes[userNo][self.node_nums[userNo] - 1] != -1:
            done = True
        else:
            done = False
            next_taskNo = self.curr_task[userNo]

            self.pres[userNo][next_taskNo] = self.pres[userNo][next_taskNo] + [-1] * (
                    10 - len(self.pres[userNo][next_taskNo]))
            self.weights[userNo][next_taskNo] = self.weights[userNo][next_taskNo] + [-1] * (
                    10 - len(self.weights[userNo][next_taskNo]))

            self.states[userNo][len(self.mecs): len(self.mecs) + 20] = self.pres[userNo][next_taskNo] + \
                                                                       self.weights[userNo][
                                                                           next_taskNo]

        r = self.get_reward3(taskNo, userNo, re[3])
        return r, done

    def judgeZero(self, array):
        count = 0
        for i in array:
            if i != 0:
                count += 1
        return count

    def step(self, action, userNo):
        policy = action + 1
        taskNo = self.curr_task[userNo]

        self.taskTypes[userNo][taskNo] = policy

        r, is_terminal = self.get_env_back(userNo)

        if is_terminal:
            # print("Current user is: " + str(userNo))
            # print("Current graph is: ")
            # print(self.graphs[userNo].printGraph(self.graphs[userNo], self.node_nums[userNo]))
            # print("Finally state is: ")
            # for i in range(self.node_nums[userNo]):
            #     print(str(self.taskTypes[userNo][i]) + ",", end=" ")
            # print("Execute time is: ")
            # print(self.endTimes[userNo])
            # for i in range(self.node_nums[userNo]):
            #     print(str(self.endTimes[userNo][i] - self.preEndTimes[userNo]) + ",", end="")
            self.preEndTimes[userNo] = self.endTimes[userNo][self.node_nums[userNo] - 1]
            # print()

        return self.states[userNo], r, is_terminal, {}, {}

    def reset(self, userNo):
        self.fix_node_num = 35
        self.node_nums[userNo] = random.randint(30, 35)
        self.localTasks[userNo] = [0, self.node_nums[userNo] - 1]
        self.mecs[0] = self.local_capacity[userNo]
        straightMec = self.straightMec[userNo] - 1
        self.mecs[1] = self.edge_capacity[straightMec]
        self.mecs[2] = self.cloud_capacity
        self.mecs[3:7] = self.velocity
        self.mecs[7] = self.straightMec[userNo]
        self.mecs[8] = self.max_task_num[straightMec]
        self.mecs[9] = self.judgeZero(self.edgeTime[straightMec])
        self.taskTypes[userNo] = [-1 for i in range(self.fix_node_num)]

        self.states[userNo] = [-1] * self.state_num

        self.graphs[userNo] = Graph()
        self.edges = self.graphs[userNo].random_graph(self.node_nums[userNo])
        self.edgeRelations[userNo] = self.graphs[userNo].initGraph(self.graphs[userNo], self.edges,
                                                                   self.node_nums[userNo])

        if userNo == 0:
            print("if i==0:")
        else:
            print("elif i== " + str(userNo), end=":")
            print()
        print(self.graphs[userNo].printGraph(self.graphs[userNo], self.node_nums[userNo]))

        self.pres[userNo] = []
        for i in range(self.node_nums[userNo]):
            i = self.graphs[userNo].getVertex(i)
            self.pres[userNo].append(self.graphs[userNo].getParents(i))
        self.weights[userNo] = []
        for i in range(self.node_nums[userNo]):
            t = []
            for s in self.pres[userNo][i]:
                t.append(self.edgeRelations[userNo][s * self.node_nums[userNo] + i])
            self.weights[userNo].append(t)

        self.pres[userNo][0] = self.pres[userNo][0] + [-1] * (10 - len(self.pres[userNo][0]))
        self.weights[userNo][0] = self.weights[userNo][0] + [-1] * (10 - len(self.weights[userNo][0]))

        self.tasks[userNo] = self.pres[userNo][0] + self.weights[userNo][0]

        self.taskTypes[userNo][0] = 1 

        self.endTimes[userNo] = np.array([0] * self.node_nums[userNo], dtype=float)
        self.endTimes[userNo][0] = self.preEndTimes[userNo]
        self.maxEnds[userNo] = np.array([0] * self.node_nums[userNo], dtype=float)

        re = getReward(0, self.tasks[userNo], self.taskTypes[userNo], self.endTimes[userNo],
                       self.localTime[userNo], self.edgeTime,
                       self.local_capacity[userNo], self.edge_capacity, self.cloud_capacity, self.velocity,
                       self.max_task_num, self.states[userNo][7])

        self.pres[userNo][1] = self.pres[userNo][1] + [-1] * (10 - len(self.pres[userNo][1]))
        self.weights[userNo][1] = self.weights[userNo][1] + [-1] * (10 - len(self.weights[userNo][1]))

        self.tasks[userNo] = self.pres[userNo][1] + self.weights[userNo][1]
        self.states[userNo] = self.mecs + self.tasks[userNo]

        self.localTime[userNo] = re[0]
        self.edgeTime = re[1]

        self.endTimes[userNo][0] = re[2]
        self.maxEnds[userNo][0] = self.endTimes[userNo][0]

        self.curr_task[userNo] = 1
        return self.states[userNo]

    def getNodeNum(self):
        return self.node_num

    def getEndTime(self):
        return self.endTime

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
