# -- coding: utf-8 --
# @Time : 2022/5/14 16:00
# @Author : Mkc
# @Email : mkc17@foxmail.com
# @Software: PyCharm

import numpy as np
import networkx as nx
import time
import random
import math
import copy
from time import strftime, localtime

def IC_model(g,S,mc,p=0.01):
    spread = []
    for i in range(mc):
        new_active, Au = S[:], S[:]
        while new_active:
            new_ones = []
            for node in new_active:
                nbs = list(set(g.neighbors(node)) - set(Au))
                for nb in nbs:
                    if random.random() <= p:
                        new_ones.append(nb)
            new_active = list(set(new_ones))
            Au += new_active
        spread.append(len(Au))
    return np.mean(spread)

#EDV
def Eval(G, S):
    Neighbors = Neighbor_Nodeset(G, S)

    fitness = len(S)
    L = list(set(Neighbors) - set(S))
    for TIME in L:
        fitness += 1 - (1 - 0.01) ** len(set(G.neighbors(TIME)) & set(S))
    return fitness

def Neighbor_nodes(G,u):
    return list(G.neighbors(u)) + [u]

def Neighbor_Nodeset(G,S):
    neighbors = [ ]
    for i in S:
        neighbors += Neighbor_nodes(G,i)
    neighbors = list(set(neighbors))
    return neighbors

def pop_init(nodes,pop,K,t,avg_d):
    P = []

    for i in range(pop):
        P_Item = []
        for Kt in range(K):
            temp = math.ceil((Kt+1) * math.exp(t * avg_d))+Kt+1
            if temp>len(nodes):
                temp = len(nodes)

            P_Item.append(nodes[random.randint(0,temp-1)])
        P.append(P_Item)

    return P

def mutation(G,P,P_remain,mu,K,t,avg_d,nodes):
    P_new = copy.deepcopy(P)

    for P_It in P_new:
        if P_new.index(P_It) == 0:
            temp = math.ceil(K * math.exp(t * avg_d))+K
            if temp <= K:
                temp += 1
            if temp>len(nodes):
                temp = len(nodes)

            ran_int = random.randint(0,K-1)

            while True:
                ran_temp = random.randint(0,temp-1)
                if nodes[ran_temp] not in P_It:
                    P_It[ran_int] = nodes[ran_temp]
                    break
            continue

        if P_new.index(P_It) < len(P_new)/2:
            temp = math.ceil(K * math.exp(t * avg_d))+K
        else:
            temp = math.ceil((K-1) * math.exp(t * avg_d))+K-1

        if temp <= K:
            temp += 1
        if temp > len(nodes):
            temp = len(nodes)

        # candidate nodes
        node_set = []
        # probability of candidate nodes being selected
        degree_pro = []

        for i in range(temp):
            node_set.append(nodes[i])
            degree_pro.append(G.degree(nodes[i]))

        degree_sum = sum(degree_pro)
        for deg in range(len(degree_pro)):
            degree_pro[deg] = degree_pro[deg] / degree_sum
        degree_pro = np.array(degree_pro)

        for index in range(len(P_It)):
            if P_It[index] not in P_remain[P_new.index(P_It)]:
                if random.random() < mu:
                    while True:
                        temp_d = np.random.choice(node_set, size=1, p=degree_pro.ravel())

                        if temp_d[0] not in P_It:
                            P_It[index] = temp_d[0]
                            break
    return P_new

def crossover(P,cr,pop):
    P_c = []
    P_remain = []

    random_int = 0
    P_c.append(P[random_int])

    P_remain.append([])
    for pop_index in range(pop):
        if pop_index == random_int:
            continue

        ind = list(set(P[random_int]) & set(P[pop_index]))
        P_remain.append(ind)
        P_1 = copy.deepcopy(P)
        P_1[random_int] = list(set(P_1[random_int]) - set(ind))
        P_1[pop_index] = list(set(P_1[pop_index]) - set(ind))

        for kk in range(len(P_1[random_int])):
            if random.random() < cr:
                ind.append(P_1[random_int][kk])
            else:
                ind.append(P_1[pop_index][kk])
        P_c.append(ind)
    return P_c,P_remain


def main():
    for address in ["musae-facebook.txt"]:
        print(address)
        for K in [50]:
            result1 = []#influence spread
            result2 = []#running time
            for A in range(2):
                print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
                G = nx.read_edgelist(address, create_using=nx.Graph())
                start_time = time.perf_counter()

                nodes = list(G.nodes)
                edges = list(G.edges)
                nodes = sorted(nodes, key=lambda x: G.degree(x), reverse=True)

                pop = 10
                t = 0.04
                avg_d = round(2*len(edges)/len(nodes),2)
                mu = 0.1
                cr = 0.6
                maxgen = 15

                #initialization
                P = pop_init(nodes,pop,1,t,avg_d)

                for KK in range(2,K+1):
                    i = 0
                    #seed addition
                    temp = math.ceil(KK * math.exp(t * avg_d)) + KK
                    if temp>len(nodes):
                        temp = len(nodes)

                    for ind in range(len(P)):
                        while True:
                            ran_index = random.randint(0,temp-1)
                            if nodes[ran_index] not in P[ind]:
                                P[ind].append(nodes[ran_index])
                                break

                    while i < maxgen:
                        P = sorted(P,key=lambda x:Eval(G,x),reverse=True)

                        # mutation&crossover
                        P_cross, P_remain = crossover(P, cr, pop)
                        P_mutation = mutation(G,P_cross, P_remain, mu, KK, t, avg_d, nodes)

                        #selection
                        for index in range(pop):
                            Inf1 = Eval(G, P_mutation[index])
                            Inf2 = Eval(G, P[index])
                            if Inf1 > Inf2:
                                P[index] = P_mutation[index]
                        i += 1

                solution = sorted(P,key=lambda x:Eval(G,x),reverse=True)[0]

                end_time = time.perf_counter()
                runningtime = end_time-start_time

                result1.append(IC_model(G, solution, 10000))
                result2.append(runningtime)

            print(result1)
            print(result2)
            print("Network:", address, "K:", K, "Influence spread:", round(np.mean(result1), 1),"Running time:",round(np.mean(result2),1))
            print()


if __name__ == '__main__':
    main()