import sys
from queue import Queue
import copy
from graph import Graph

c_size = [107, 150, 140, 220, 131, 128, 434, 378, 352, 392, 442, 297, 13, 366, 384, 48, 112, 200, 362, 490, 342, 358,
          317, 138, 467, 337, 261, 442, 135, 383, 93, 133, 266, 221, 448, 381, 473, 130, 392, 135, 315, 227, 284, 107,
          154, 429, 63, 195, 448, 365]


def getWaitTime(currType, actualType, taskType, endTime, localTime, edgeTime, graphInfo, mecNo, velocity):
    parents = graphInfo[0:10]
    taskArriveTime, waitTime = 0, 0

    for i, parent in enumerate(parents):
        if parent == -1:
            break
        currEndTime = endTime[parent]
        parentType = taskType[parent]
        transTime = 0
        if parentType != currType:
            weight = graphInfo[10:20][i]
            if (parentType == 1 and currType == mecNo + 1) or (
                    parentType == mecNo + 1 and currType == 1):
                transTime = weight / velocity[0]
            elif (parentType == 1 and currType == 5) or (parentType == 5 and currType == 1):
                transTime = min(weight / velocity[2], weight / velocity[0] + weight / velocity[3])
            elif (parentType == 1 and currType in [2, 3, 4]) or (
                    currType == 1 and parentType in [2, 3, 4]):
                t = [parentType, currType]
                t = max(t)
                transTime = weight / velocity[0] + (weight / velocity[1]) * abs(t - (mecNo + 1))
            elif (parentType in [2, 3, 4] and currType in [2, 3, 4]) or (
                    parentType in [2, 3, 4] and currType in [2, 3, 4]):
                transTime = (weight / velocity[1]) * abs(parentType - currType)
            elif (parentType in [2, 3, 4] and currType == 5) or (
                    parentType == 5 and currType in [2, 3, 4]):
                transTime = weight / velocity[3]
        taskArriveTime = max(taskArriveTime, currEndTime + transTime)

        if currType == 1:
            if taskArriveTime > localTime:
                localTime = 0
            waitTime = localTime
        elif currType in [2, 3, 4]:
            for j in range(0, len(edgeTime[currType - 2])):
                if taskArriveTime > edgeTime[currType - 2][j]:
                    edgeTime[currType - 2][j] = 0
            waitTime = min(edgeTime[currType - 2])
        else:
            waitTime = 0

    return max(waitTime, taskArriveTime)


def getExecuteTime(id, type, capacity):
    return c_size[id] / capacity[int(type) - 1]


def codeOffloading(num, graphInfo, taskType, endTime, localTime, edgeTime, localCapacity, edgeCapacity, cloudCapacity,
                   velocity, maxNum, mecNo):
    curr = num
    capacity = [localCapacity] + edgeCapacity + [cloudCapacity]
    allWaitTime = []
    if curr == 0:
        executeTime = getExecuteTime(0, 1, capacity)
        taskType[0] = 1
        currTime = localTime = endTime[0] = executeTime + endTime[0]
    else:
        currType = taskType[curr]

        for i in range(1, 6):
            allWaitTime.append(
                getWaitTime(i, currType, taskType, endTime, localTime, edgeTime, graphInfo, mecNo, velocity))

        waitTime = allWaitTime[currType - 1]

        executeTime = getExecuteTime(curr, currType, capacity)

        currTime = waitTime + executeTime

        endTime[curr] = currTime

        if currType == 1:
            localTime = currTime
        elif currType in [2, 3, 4]:
            t = currType - 2
            if len(edgeTime[t]) == maxNum[t]:
                edgeTime[t].remove(min(edgeTime[t]))
            edgeTime[t].append(currTime)

    return localTime, edgeTime, currTime, allWaitTime


def getReward(i, graphInfo, taskType, endTime, localTime, edgeTime, localCapacity, edgeCapacity, cloudCapacity,
              velocity, maxNum, mecNo):
    return codeOffloading(i, graphInfo, taskType, endTime, localTime, edgeTime, localCapacity, edgeCapacity,
                          cloudCapacity, velocity, maxNum, mecNo)
