from pyspark import SparkConf, SparkContext
import itertools
import sys
import time
from collections import defaultdict


def Girvan_Newman(root, vertices, graph):
    tree = {0: root}
    path_num = {root: 1}
    nodes_level = graph[root]
    children = {root: nodes_level}
    level = 1
    parents = defaultdict(set)
    visited = set()
    visited.add(root)

    for v in nodes_level:
        parents[v].add(root)

    while nodes_level != set():
        tree[level] = nodes_level
        level += 1
        for node in nodes_level:
            visited.add(node)
        next_level = set()
        for node in nodes_level:
            new_node = set()
            for new in graph[node]:
                if new not in visited:
                    new_node.add(new)
                    parents[new].add(node)
                    next_level.add(new)
            children[node] = new_node
            cur_node_parents = parents[node]
            # print(node, cur_node_parents)
            path_num[node] = 0
            if len(cur_node_parents) != 0:
                for parent in cur_node_parents:
                    # print(node, parent, cur_node_parents)
                    path_num[node] += path_num[parent]
            else:
                path_num[node] = 1
        nodes_level = next_level

    node_credit = defaultdict(float)
    edge_contribution = {}
    for v in vertices:
        node_credit[v] = 1
    while level > 1:
        for v in tree[level - 1]:
            total = path_num[v]
            for parent in parents[v]:
                edge = tuple(sorted((v, parent)))
                edge_contribution[edge] = node_credit[v] * (path_num[parent] / total)
                node_credit[parent] += edge_contribution[edge]
        level -= 1
    return [(k, v) for k, v in edge_contribution.items()]


def get_all_community(vertices, graph):
    communities = []
    visited = set()
    for v in vertices:
        if v in visited:
            continue
        visited.add(v)
        cur_step = graph[v]
        cur_community = [v]
        if len(cur_step) == 0:
            communities.append(cur_community)
            continue
        while cur_step:
            next_step = set()
            for node in cur_step:
                if node in visited:
                    continue
                visited.add(node)
                cur_community.append(node)
                node_next = graph[node]
                for next_n in node_next:
                    if next_n not in visited:
                        next_step.add(next_n)
            cur_step = next_step
        communities.append(cur_community)
    return communities


def calculate_modularity(communities, m, A, degree):
    Q = 0.0
    for community in communities:
        cur_Q = 0.0
        for i in community:
            for j in community:
                cur_Q += A[(i, j)] - (degree[i] * degree[j]) / (2 * m)
        Q += cur_Q
    return Q / (2 * m)


def find_best_community(vertices, edges, graph, vertex_rdd):
    best_Q = -1
    best_community = []
    m = len(edges) / 2
    edge_left = m
    degree = {}
    for key, value in graph.items():
        degree[key] = len(value)

    A = defaultdict(float)
    for edge in edges:
        A[(edge[0], edge[1])] = 1
        A[(edge[1], edge[0])] = 1

    while edge_left != 0:
        betweenness = vertex_rdd.map(lambda vertex: Girvan_Newman(vertex, vertices, graph)). \
            flatMap(lambda x: [p for p in x]). \
            reduceByKey(lambda a, b: a + b). \
            map(lambda x: (x[0], x[1] / 2)). \
            sortBy(lambda x: (-x[1], x[0])). \
            collect()
        highest = betweenness[0][1]
        for line in betweenness:
            if line[1] == highest:
                node1 = line[0][0]
                node2 = line[0][1]
                graph[node1].remove(node2)
                graph[node2].remove(node1)
                edge_left -= 1
            else:
                break
        cur_community = get_all_community(vertices, graph)
        cur_Q = calculate_modularity(cur_community, m, A, degree)
        if cur_Q > best_Q:
            best_Q = cur_Q
            best_community = cur_community
    return best_community


def task2(filter_threshold, input_file_path, betweenness_output_file_path, community_output_file_path):
    start = time.time()
    conf = SparkConf().setAppName("DSCI553").setMaster('local[*]')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    rdd = sc.textFile(input_file_path)
    header = rdd.first()
    ub_rdd = rdd.filter(lambda x: x != header).map(lambda x: x.split(',')).cache()

    uid_rdd = ub_rdd.map(lambda x: x[0])
    distinct_user = uid_rdd.distinct().collect()
    user_map = ub_rdd.groupByKey().map(lambda x: (x[0], list(x[1]))).collectAsMap()

    vertices = set()
    edges = set()
    for pair in itertools.combinations(distinct_user, 2):
        if len(set(user_map[pair[0]]).intersection(set(user_map[pair[1]]))) >= int(filter_threshold):
            vertices.add(pair[0])
            vertices.add(pair[1])
            edges.add((pair[1], pair[0]))
            edges.add(pair)

    graph = defaultdict(set)
    for pair in edges:
        graph[pair[0]].add(pair[1])

    vertex_rdd = sc.parallelize(vertices)
    betweenness = vertex_rdd.map(lambda vertex: Girvan_Newman(vertex, vertices, graph)). \
        flatMap(lambda x: [p for p in x]). \
        reduceByKey(lambda a, b: a + b). \
        map(lambda x: (x[0], x[1] / 2)). \
        sortBy(lambda x: (-x[1], x[0])). \
        collect()

    with open(betweenness_output_file_path, 'w') as f:
        for p in betweenness:
            f.write(str(p[0]) + "," + str(round(float(p[1]), 5)) + "\n")

    community_list = find_best_community(vertices, edges, graph, vertex_rdd)
    best_communities = sc.parallelize(community_list). \
        map(lambda x: sorted(x)). \
        sortBy(lambda x: (len(x), x)). \
        collect()

    with open(community_output_file_path, 'w') as f:
        for community in best_communities:
            f.write(str(community)[1:-1] + "\n")
    print("Duration: ", time.time() - start)


filter_threshold, input_file_path, betweenness_output_file_path, community_output_file_path = int(sys.argv[1]), \
                                                                                              sys.argv[2], sys.argv[3], \
                                                                                          sys.argv[4]
task2(filter_threshold, input_file_path, betweenness_output_file_path, community_output_file_path)
