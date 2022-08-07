import re
from socket import timeout
import pandas as pd
import numpy as np
import pulp
import math
import os

import networkx as nx
import matplotlib.pyplot as plt


class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Edge:
    def __init__(self, u, v, c):
        self.u = u # 始点
        self.v = v # 終点
        self.c = c # コスト
    
    def __str__(self):
        return f'(u, v, c) = ({self.u}, {self.v}, {self.c})'
    
    def __repr__(self) -> str:
        return f'(u, v, c) = ({self.u}, {self.v}, {self.c})'

    def __eq__(self, other):
        return self.u == other.u and self.v == other.v and self.c == other.c

    def __hash__(self):
        return hash(self.u * 1000000 + self.v * 10000 + self.c)



def l2norm(p0: tuple[int, int], p1: tuple[int, int]) -> int:
    return math.hypot(p0[0] - p1[0], p0[1] - p1[1])

# ランダムなTSPを作成
def make_tsp(num_verts: int, num_neighbours: int, seed: int = 0) -> tuple[list[Vector2D], list[Edge]]:
    np.random.seed(seed=seed)
    x_coords = np.random.randint(0, 1000, num_verts)
    y_coords = np.random.randint(0, 1000, num_verts)
    coords = list(zip(x_coords, y_coords))
    coords[0] = (500, 500)

    edges = []
    for i in range(num_verts):
        edges_with_cost: list[Edge] = [Edge(i, j, math.floor(l2norm(coords[i], coords[j]))) for j in range(num_verts) if i != j]
        edges_with_cost.sort(key=lambda e: e.c)
        edges.extend(edges_with_cost[:min(len(edges_with_cost), num_neighbours)])
    
    return coords, edges


def add_mtz_constraints(problem, x, verts, edges, edge_index, edges_from, edges_to, num_salesmen):
    num_verts = len(verts)
    num_edges = len(edges)
    u = []
    for k in range(num_salesmen):
        kk_edges = num_edges * k
        kk_verts = num_verts * k
        u.extend([pulp.LpVariable(f'u_{k}_{i}', 0, num_verts - 1) for i in range(num_verts)])

        for e in edges:
            if e.v != 0:
                problem += u[kk_verts + e.v] + num_verts * (1 - x[kk_edges + edge_index[(e.u, e.v)]]) >= u[kk_verts + e.u] + 1
        problem += u[kk_verts + 0] == 0

def add_flow_constraints(problem, x, verts, edges, edge_index, edges_from, edges_to, num_salesmen):
    num_verts = len(verts)
    num_edges = len(edges)
    y = []
    for k in range(num_salesmen):
        kk = num_edges * k
        y.extend([pulp.LpVariable(f'y_{k}_{edge_index[(e.u, e.v)]}', 0, num_verts - 1) for e in edges])

        for e in edges:
            problem += y[kk + edge_index[(e.u, e.v)]] <= (num_verts - 1) * x[kk + edge_index[(e.u, e.v)]]
        for i in range(1, num_verts):
            problem += pulp.lpSum(y[kk + edge_index[(i, j)]] for j in edges_from[i]) == pulp.lpSum(y[kk + edge_index[(j, i)]] for j in edges_to[i]) + 1  
        
        # problem += pulp.lpSum(y[kk + edge_index[(j, 0)]] for j in edges_to[0]) - (num_verts - 1) == pulp.lpSum(y[kk + edge_index[(0, j)]] for j in edges_from[0])
        problem += pulp.lpSum(y[kk + edge_index[(0, j)]] for j in edges_from[0]) == 0

def add_flow_constraints(problem, x, verts, edges, edge_index, edges_from, edges_to, num_salesmen):
    num_verts = len(verts)
    num_edges = len(edges)

    y = [pulp.LpVariable(f'y_{k}_{edge_index[(e.u, e.v)]}', 0, num_verts - 1) for k in range(num_salesmen) for e in edges]

    for e in edges:
        problem += pulp.lpSum(y[num_edges * k + edge_index[(e.u, e.v)]] for k in range(num_salesmen)) <= (num_verts - 1) * pulp.lpSum(x[num_edges * k + edge_index[(e.u, e.v)]] for k in range(num_salesmen))
    for i in range(1, num_verts):
        problem += pulp.lpSum(y[num_edges * k + edge_index[(i, j)]] for k in range(num_salesmen) for j in edges_from[i]) == pulp.lpSum(y[num_edges * k + edge_index[(j, i)]] for k in range(num_salesmen) for j in edges_to[i]) + 1  
    
    # problem += pulp.lpSum(y[kk + edge_index[(j, 0)]] for j in edges_to[0]) - (num_verts - 1) == pulp.lpSum(y[kk + edge_index[(0, j)]] for j in edges_from[0])
    problem += pulp.lpSum(y[num_edges * k + edge_index[(0, j)]] for k in range(num_salesmen) for j in edges_from[0]) == 0

def draw_route_from_sol(sol_file_path: str, verts, edges):
    num_verts = len(verts)
    result = pd.read_csv(sol_file_path)
    result2 = result[result['variable'].str.startswith('x_') & (result['value'] == 1)]

    edges_df = result2['variable'].str.split('_', expand=True)
    edges_df[1] = edges_df[1].astype(int)
    edges_df[2] = edges_df[2].astype(int)
    c = ['r', 'g', 'b', 'y', 'c']
    path_edges = [(edges[r[2]].u, edges[r[2]].v, {'color': c[r[1]]}) for _, r in edges_df.iterrows()]
    edges_df[1] = edges_df[1].astype(int)

    print(f'weight: {sum([edges[i].c for i in edges_df[2].astype(int)])}')
    print(edges_df.shape)

    # グラフの描画
    G = nx.DiGraph()
    G.add_nodes_from(list(range(num_verts)))
    G.add_edges_from(path_edges)

    edge_color = [e['color'] for e in G.edges.values()]

    pos = {i : verts[i] for i in range(num_verts)}
    fig = plt.figure()
    nx.draw_networkx(G, pos, alpha=0.5, node_size=10, with_labels=False, edge_color=edge_color)
    plt.axis('off')
    fig.savefig('./test.png')


def main():
    # TSPの問題を生成
    num_verts = 200
    num_neighbours = 15
    seed = 1
    verts, edges = make_tsp(num_verts=num_verts, num_neighbours=num_neighbours, seed=seed)


    # エッジを双方向に張る
    edges_set = set(edges)
    for e in edges_set:
        e_rev = Edge(e.v, e.u, e.c)
        if e_rev not in edges_set:
            edges.append(e_rev)

    # draw_route_from_sol('model.sol.csv', verts, edges)
    # return 

    edge_index: dict[Edge, int] = {
        (e.u, e.v) : i for i, e in enumerate(edges)
    }
    num_edges = len(edges)

    edges_from = [[e.v for e in edges if e.u == u] for u in range(num_verts)]
    edges_to = [[e.u for e in edges if e.v == u] for u in range(num_verts)]


    problem = pulp.LpProblem("TSP", pulp.LpMinimize)
    # problem = pulp.LpProblem("TSP", pulp.LpMaximize)

    num_salesmen = 4
    x = [pulp.LpVariable(f'x_{k}_{edge_index[(e.u, e.v)]}', cat=pulp.const.LpBinary) for k in range(num_salesmen) for e in edges]


    # 目的関数の定義
    # problem += pulp.lpSum(e.c * x[num_edges * k + edge_index[(e.u, e.v)]] for k in range(num_salesmen) for e in edges)
    problem += -pulp.lpSum(x[num_edges * k + edge_index[(e.u, e.v)]] for k in range(num_salesmen) for e in edges)

    for k in range(num_salesmen):
        kk = num_edges * k
        for e in edges:
            problem += x[kk + edge_index[(e.u, e.v)]] + x[kk + edge_index[(e.v, e.u)]] <= 1

    # 制約条件の定義
    for k in range(num_salesmen):
        kk = num_edges * k
        for i in range(0, num_verts):
            problem += pulp.lpSum(x[kk + edge_index[(j, i)]] for j in edges_to[i]) == pulp.lpSum(x[kk + edge_index[(i, j)]] for j in edges_from[i])
            problem += pulp.lpSum(x[kk + edge_index[(j, i)]] for j in edges_to[i]) <= 1 
            problem += pulp.lpSum(x[kk + edge_index[(i, j)]] for j in edges_from[i]) <= 1
        
    # 2opt 制約
    # for i in range(num_edges):
    #     for j in range(i + 1, num_edges):
    #         e0 = edges[i]
    #         e1 = edges[j]
    #         if e0.u == e1.v or e1.u == e0.v:
    #             continue

    #         if (e0.u, e1.u) not in edge_index or (e0.v, e1.v) not in edge_index:
    #             continue
    #         len_prev = edges[edge_index[(e0.u, e0.v)]].c + edges[edge_index[(e1.u, e1.v)]].c
    #         len_next = edges[edge_index[(e0.u, e1.u)]].c + edges[edge_index[(e0.v, e1.v)]].c
    #         if len_next > len_prev:
    #             continue
    #         problem += pulp.lpSum(x[num_edges * k + edge_index[(e0.u, e0.v)]] + x[num_edges * k + edge_index[(e1.u, e1.v)]] for k in range(num_salesmen)) <= 1


    add_mtz_constraints(problem, x, verts, edges, edge_index, edges_from, edges_to, num_salesmen)
    # add_flow_constraints(problem, x, verts, edges, edge_index, edges_from, edges_to, num_salesmen)

    # 労働時間の制約
    for k in range(num_salesmen):
        kk = num_edges * k
        problem += pulp.lpSum(e.c * x[kk + edge_index[(e.u, e.v)]] for e in edges) <= 2000

    # 人がかぶらないようにする
    for i in range(1, num_verts):
        problem += pulp.lpSum(x[k * num_edges + edge_index[(i, j)]] for k in range(num_salesmen) for j in edges_from[i]) <= 1
        problem += pulp.lpSum(x[k * num_edges + edge_index[(j, i)]] for k in range(num_salesmen) for j in edges_to[i]) <= 1

    # 解く
    num_threads = 1
    max_seconds = 600
    # solver_path = "C:/home/bin/Cbc-2.10.5-x86_64-w64-mingw32/bin/cbc.exe"
    # solver_path = "C:/Users/kouya/Cbc-2.10.5-x86_64-w64-mingw32/bin/cbc.exe"
    solver_path = "C:/home/bin/Cbc-releases.2.10.7-w64-msvc16-md/bin/cbc.exe"
    problem.writeMPS('./test.mps')



    result = problem.solve(pulp.COIN_CMD(path=solver_path, threads=num_threads, timeLimit=max_seconds, msg=True))

    path_edges = [(e.u, e.v) for k in range(num_salesmen) for e in edges if pulp.value(x[num_edges * k + edge_index[(e.u, e.v)]]) == 1]

    # グラフの描画
    G = nx.DiGraph()
    G.add_nodes_from(list(range(num_verts)))
    G.add_edges_from(path_edges)

    pos = {i : verts[i] for i in range(num_verts)}
    fig = plt.figure()
    nx.draw_networkx(G, pos, alpha=0.5)
    plt.axis('off')
    fig.savefig('./test.png')

if __name__ == '__main__':
    main()