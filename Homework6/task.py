import os
import sys
import time
import copy
import random
import itertools
import numpy as np
from sklearn.cluster import KMeans


def init_dict(pair, data_dict_reversed):
    return {
        "N": [data_dict_reversed[tuple(pair[1])]],
        "SUM": pair[1],
        "SUMSQ": pair[1] ** 2
    }


def add_pair(collection, pair0, pair1, data_dict_reversed):
    collection[pair0]["N"].append(data_dict_reversed[tuple(pair1)])
    collection[pair0]["SUM"] += pair1
    collection[pair0]["SUMSQ"] += pair1 ** 2


def get_K_means(RS):
    if len(RS) == 1:
        return KMeans(n_clusters=1).fit(RS)
    else:
        return KMeans(n_clusters=len(RS) - 1).fit(RS)


def intermediate_result(DS, CS, RS, result, round):
    DS_N = sum([len(DS[cluster]["N"]) for cluster in DS])
    CS_N = sum([len(CS[cluster]["N"]) for cluster in CS])
    result.append(
        "Round " + str(round) + ": " + str(DS_N) + "," + str(len(CS)) + "," + str(CS_N) + "," + str(len(RS)) + "\n")


def mahalanobis_distance(point, cluster):
    c = cluster["SUM"] / len(cluster["N"])
    sigma = cluster["SUMSQ"] / len(cluster["N"]) - (cluster["SUM"] / len(cluster["N"])) ** 2
    Sum = (point - c) / sigma
    return np.dot(Sum, Sum) ** (1 / 2)


def mahalanobis_distance_cluster(cluster_1, cluster_2):
    center_1 = cluster_1["SUM"] / len(cluster_1["N"])
    center_2 = cluster_2["SUM"] / len(cluster_1["N"])
    sigma_1 = cluster_1["SUMSQ"] / len(cluster_1["N"]) - (cluster_1["SUM"] / len(cluster_1["N"])) ** 2
    sigma_2 = cluster_2["SUMSQ"] / len(cluster_2["N"]) - (cluster_2["SUM"] / len(cluster_2["N"])) ** 2
    sum_1 = (center_1 - center_2) / sigma_1
    sum_2 = (center_1 - center_2) / sigma_2
    return min(np.dot(sum_1, sum_1) ** (1 / 2), np.dot(sum_2, sum_2) ** (1 / 2))


def task(input_file, n_cluster, output_file):
    start_time = time.time()
    n_cluster = int(n_cluster)
    result = ["The intermediate results:\n"]
    # Step 1. Load 20% of the data randomly.
    with open(input_file) as f:
        data = f.readlines()
    data = list(map(lambda x: x.strip("\n").split(','), data))
    data = [(int(line[0]), tuple([float(feature) for feature in line[2:]])) for line in data]
    data_dict = dict(data)
    data_dict_reversed = dict(zip(list(data_dict.values()), list(data_dict.keys())))
    data = list(map(lambda x: np.array(x), list(data_dict.values())))
    random.shuffle(data)
    cluster_len = round(len(data) / 5)
    cluster_data = data[0: cluster_len]
    # Step 2. Run K-Means (e.g., from sklearn) with a large K (e.g., 5 times of the number of the input clusters)
    # on the data in memory using the Euclidean distance as the similarity measurement.
    k_means = KMeans(n_clusters=n_cluster * 10).fit(cluster_data)

    
    # Step 3. In the K-Means result from Step 2, move all the clusters that contain only one point to RS (outliers).
    cluster_result_count = {}
    for label in k_means.labels_:
        cluster_result_count[label] = cluster_result_count.get(label, 0) + 1

    RS = []
    RS_idx = []
    for key in cluster_result_count.keys():
        if cluster_result_count[key] == 1:
            for idx, label in enumerate(k_means.labels_):
                if key == label:
                    RS_idx.append(idx)
                    break

    RS += [cluster_data[i] for i in RS_idx]
    for idx in reversed(sorted(RS_idx)):
        cluster_data.pop(idx)
    # Step 4. Run K-Means again to cluster the rest of the data points with K = the number of input clusters.
    k_means = KMeans(n_clusters=n_cluster).fit(cluster_data)
    # Step 5. Use the K-Means result from Step 4 to generate the DS clusters (i.e., discard their points and generate statistics).
    DS = {}
    cluster_pair = tuple(zip(k_means.labels_, cluster_data))

    for pair in cluster_pair:
        if pair[0] not in DS:
            DS[pair[0]] = init_dict(pair, data_dict_reversed)
            continue
        add_pair(DS, pair[0], pair[1], data_dict_reversed)
    # Step 6. Run K-Means on the points in the RS with a large K (e.g., 5 times of the number
    # of the input clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point).
    if len(RS) > 0:
        k_means = get_K_means(RS)

        cluster_result_count = {}
        for label in k_means.labels_:
            cluster_result_count[label] = cluster_result_count.get(label, 0) + 1
        RS_temp_idx = [k for k in cluster_result_count.keys() if cluster_result_count[k] == 1]
        if len(RS_temp_idx) > 0:
            RS_idx = [list(k_means.labels_).index(k) for k in RS_temp_idx]
        cluster_pair = tuple(zip(k_means.labels_, RS))
        CS = {}
        for pair in cluster_pair:
            if pair[0] not in RS_temp_idx:
                if pair[0] not in CS:
                    CS[pair[0]] = init_dict(pair, data_dict_reversed)
                    continue
                add_pair(CS, pair[0], pair[1], data_dict_reversed)
        RS_update = [RS[i] for i in reversed(sorted(RS_idx))]
        RS = copy.deepcopy(RS_update)

    intermediate_result(DS, CS, RS, result, 1)

    for j in range(4):
        # Step 7. Load another 20% of the data randomly.
        if j == 3:
            cluster_data = data[cluster_len * 4:]
        else:
            cluster_data = data[cluster_len * (j + 1): cluster_len * (j + 2)]
        # Step 8. For the new points, compare them to each of the DS using
        # the Mahalanobis Distance and assign them to the nearest DS clusters if the distance is < 2 𝑑.
        DS_idx = set()
        for i in range(len(cluster_data)):
            point = cluster_data[i]
            min_dist = sys.maxsize
            cst = 0
            for cluster in DS:
                dist = mahalanobis_distance(point, DS[cluster])
                if min_dist > dist:
                    min_dist = dist
                    cst = cluster
            if min_dist < 2 * (len(point) ** (1 / 2)):
                DS_idx.add(i)
                add_pair(DS, cst, point, data_dict_reversed)
        #   Step 9. For the new points that are not assigned to DS clusters,
        #   using the Mahalanobis Distance and assign the points to the nearest CS clusters if the distance is < 2 𝑑
        if len(CS) > 0:
            CS_idx = set()
            for i in range(len(cluster_data)):
                if i in DS_idx:
                    continue
                point = cluster_data[i]
                min_dist = sys.maxsize
                cst = 0
                for cluster in CS:
                    dist = mahalanobis_distance(point, CS[cluster])
                    if min_dist > dist:
                        min_dist = dist
                        cst = cluster
                if min_dist < 2 * (len(point) ** (1 / 2)):
                    CS_idx.add(i)
                    add_pair(CS, cst, point, data_dict_reversed)
        # Step 10. For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.
        for i in range(len(cluster_data)):
            if i not in DS_idx and i not in CS_idx:
                RS.append(cluster_data[i])
        # Step 11. Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters)
        # to generate CS (clusters with more than one points) and RS (clusters with only one point).
        if len(RS) > 0:
            k_means = get_K_means(RS)
        CS_set = set(CS.keys())
        RS_set = set(k_means.labels_)
        intersection = CS_set.intersection(RS_set)
        union = CS_set.union(RS_set)
        change = dict()
        for intersect in intersection:
            while True:
                random_int = random.randint(100, len(cluster_data))
                if random_int not in union:
                    break
            change[intersect] = random_int
            union.add(random_int)
        new_labels = list(k_means.labels_)
        for i in range(len(new_labels)):
            if new_labels[i] in change:
                new_labels[i] = change[new_labels[i]]
        cluster_result_count = dict()
        for label in new_labels:
            cluster_result_count[label] = cluster_result_count.get(label, 0) + 1
        RS_temp_idx = [k for k in cluster_result_count if cluster_result_count[k] == 1]
        RS_idx = [new_labels.index(k) for k in RS_temp_idx]
        cluster_pair = tuple(zip(new_labels, RS))
        for pair in cluster_pair:
            if pair[0] in RS_temp_idx:
                continue
            if pair[0] not in CS:
                CS[pair[0]] = init_dict(pair, data_dict_reversed)
                continue
            add_pair(CS, pair[0], pair[1], data_dict_reversed)
        RS_update = [RS[i] for i in reversed(sorted(RS_idx))]
        RS = copy.deepcopy(RS_update)
        # Step 12. Merge CS clusters that have a Mahalanobis Distance < 2 𝑑.
        while True:
            original_cluster = set(CS.keys())
            for compare in list(itertools.combinations(list(CS.keys()), 2)):
                if mahalanobis_distance_cluster(CS[compare[0]], CS[compare[1]]) < 2 * (
                        len(CS[compare[0]]["SUM"]) ** (1 / 2)):
                    CS[compare[0]]["N"] = CS[compare[0]]["N"] + CS[compare[1]]["N"]
                    CS[compare[0]]["SUM"] += CS[compare[1]]["SUM"]
                    CS[compare[0]]["SUMSQ"] += CS[compare[1]]["SUMSQ"]
                    CS.pop(compare[1])
                    break
            new_cluster = set(CS.keys())
            if new_cluster == original_cluster:
                break
        CS_cluster = list(CS.keys())
        if j == 3 and len(CS) > 0:
            for cluster_CS in CS_cluster:
                min_dist = sys.maxsize
                cst = 0
                for cluster in DS:
                    dist = mahalanobis_distance_cluster(CS[cluster_CS], DS[cluster])
                    if min_dist > dist:
                        min_dist = dist
                        cst = cluster
                if min_dist < 2 * len(CS[cluster_CS]["SUM"]) ** (1 / 2):
                    DS[cst]["N"] = DS[cst]["N"] + CS[cluster_CS]["N"]
                    DS[cst]["SUM"] += CS[cluster_CS]["SUM"]
                    DS[cst]["SUMSQ"] += CS[cluster_CS]["SUMSQ"]
                    CS.pop(cluster_CS)

        intermediate_result(DS, CS, RS, result, j + 2)
    result.append("\nThe clustering results:\n")
    print(result)
    for cluster in DS:
        DS[cluster]["N"] = set(DS[cluster]["N"])
    for cluster in CS:
        CS[cluster]["N"] = set(CS[cluster]["N"])
    Rs_set = set(data_dict_reversed[tuple(point)] for point in RS)
    for p in range(len(data_dict)):
        if p in RS_set:
            result.append(str(p) + ",-1\n")
        else:
            for cluster in DS:
                if p in DS[cluster]["N"]:
                    result.append(str(p) + "," + str(cluster) + "\n")
                    break
            for cluster in CS:
                if p in CS[cluster]["N"]:
                    result.append(str(p) + ",-1\n")
                    break

    with open(output_file, 'w') as f:
        for i in range(len(result)):
            f.write(result[i])

    print("Duration: " + str(time.time() - start_time))


input_file, n_cluster, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
task(input_file, n_cluster, output_file)
