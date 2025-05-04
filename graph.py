from vertex import Vertex
import random
import generate
import numpy as np
from random import shuffle as sl
from random import randint as rd

c_size = [656, 402, 385, 232, 385, 240, 912, 340, 584, 481, 812, 248, 565, 766, 969, 876, 212, 799, 252, 915]


class Graph:
    def __init__(self):
        self.verList = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.verList[key] = newVertex
        return newVertex

    def getVertex(self, n):
        if n in self.verList:
            return self.verList[n]
        else:
            return None

    def getParents(self, n):
        parents = []
        if self.verList.__contains__(n.id):
            for i in range(0, self.numVertices):
                for nbr in self.getVertex(i).getConnections():
                    if n == nbr:
                        parents.append(self.getVertex(i).getId())
            return parents
        else:
            return None

    def getSucc(self, n):
        vertex = self.getVertex(n)
        succ = vertex.getConnections()
        re = []
        for s in succ:
            re.append(s.getId())
        return re

    def getNodeNums(self):
        return len(self.verList)

    def __contains__(self, n):
        return n in self.verList

    def addEdge(self, f, t, cost):
        if f not in self.verList:
            nv = self.addVertex(f)
        if t not in self.verList:
            nv = self.addVertex(t)
        self.verList[f].addNeighbor(self.verList[t], cost)

    def getVertices(self):
        return self.verList.keys()

    def __iter__(self):
        return iter(self.verList.values())

    def random_graph(self, n):
        def prob_value(p):
            q = int(n * p)
            l = [1] * q + [0] * (n - q)
            item = random.sample(l, 1)[0]
            return item

        into_degree = [0] * n
        out_degree = [0] * n
        edges = []

        for i in range(n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                prob = prob_value(0.5)
                if prob:
                    if out_degree[i] < 2 and into_degree[j] < 2:
                        edges.append((i, j))
                        into_degree[j] += 1
                        out_degree[i] += 1

        for node, id in enumerate(into_degree):
            if node != 0:
                if id == 0:
                    edges.append((0, node))
                    out_degree[0] += 1
                    into_degree[node] += 1

        for node, od in enumerate(out_degree):
            if node != n - 1:
                if od == 0:
                    edges.append((node, n - 1))
                    out_degree[node] += 1
                    into_degree[n - 1] += 1
        return edges

    def initGraph(self, graph, edges, nodes):
        input_edge = []
        for edge in edges:
            weight = random.randint(400, 1000)
            graph.addEdge(edge[0], edge[1], weight)
        for i in range(0, nodes):
            for j in range(0, nodes):
                if (i, j) in edges:
                    vertex = graph.getVertex(i)
                    other_vertex = graph.getVertex(j)
                    input_edge.append(vertex.getWeight(other_vertex))
                else:
                    input_edge.append(0)
        return input_edge

    def getParentsAndWeights(self, graph, vertex_id):
        parents = graph.getParents(vertex_id)
        weights = []
        vertex = graph.getVertex(vertex_id)
        for parent in parents:
            weights.append(vertex.getWeight(parent))
        return parents, weights

    def getRelation(self, graph, nodes):
        input_edge, edges = [], []
        for i in range(0, nodes):
            vertex = graph.getVertex(i)
            conn = vertex.getConnections()
            for con in conn:
                edges.append((vertex.getId(), con.getId()))

        for i in range(0, nodes):
            for j in range(0, nodes):
                if (i, j) in edges:
                    vertex = graph.getVertex(i)
                    other_vertex = graph.getVertex(j)
                    input_edge.append(vertex.getWeight(other_vertex))
                else:
                    input_edge.append(0)
        return input_edge

    def printGraph(self, graph, nodes):
        for i in range(nodes):
            vertex = graph.getVertex(i)
            con = vertex.getConnections()
            for c in con:
                print("graph.addEdge(" + str(vertex.getId()) + "," + str(c.getId()) + "," + str(
                    vertex.getWeight(c)) + ")")
