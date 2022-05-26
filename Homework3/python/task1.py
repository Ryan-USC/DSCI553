from pyspark import SparkConf, SparkContext
import time
import itertools
import random
import sys

hash_size = 60
bands = 30
row = 2
a = random.sample(range(1, 3000), hash_size)
b = random.sample(range(1, 3000), hash_size)


def hashing(x, aVal, bVal, num):
    res = -1
    for user in x:
        if res == -1:
            res = (aVal * user + bVal) % num
        else:
            res = min(res, (aVal * user + bVal) % num)
    return res


def band_operation(x):
    res = []
    for i in range(bands):
        res.append(((i, tuple(x[1][i * row: (i + 1) * row])), [x[0]]))
    return res


def sort_operation(pair):
    pair_list = list(pair)
    pair_list.sort()
    return tuple(pair_list)


def jaccard(business, charMatrix):
    set1 = set(charMatrix[business[0]])
    set2 = set(charMatrix[business[1]])
    similarity = len(set1.intersection(set2)) / len(set1.union(set2))
    return business[0], business[1], similarity


def task1(input_file_name, output_file_name):
    start = time.time()
    conf = SparkConf().setAppName("DSCI553").setMaster('local[*]')
    sc = SparkContext(conf=conf)

    rdd = sc.textFile(input_file_name)
    header = rdd.first()
    data = rdd.filter(lambda x: x != header).map(lambda x: x.split(',')).map(lambda x: (x[1], x[0]))

    userRdd = data.map(lambda x: x[1]).distinct().zipWithIndex()
    userNum = userRdd.count()
    userDict = userRdd.collectAsMap()

    matrix = data.map(lambda x: (x[0], userDict[x[1]])).groupByKey().map(lambda x: (x[0], list(x[1]))).sortByKey()
    charMatrix = matrix.collectAsMap()
    signature_matrix = matrix.map(lambda x: (x[0], [hashing(x[1], a[i], b[i], userNum) for i in range(hash_size)]))
    candidate_pair = signature_matrix.flatMap(lambda x: band_operation(x)). \
        reduceByKey(lambda x, y: x + y). \
        reduceByKey(lambda x, y: x + y). \
        filter(lambda x: len(x[1]) > 1). \
        flatMap(lambda pair: list(itertools.combinations(pair[1], 2))). \
        map(lambda pair: sort_operation(pair)).distinct()

    final_result = candidate_pair.map(lambda business: jaccard(business, charMatrix)). \
        filter(lambda x: x[2] >= 0.5). \
        sortBy(lambda x: (x[0], x[1]))
    with open(output_file_name, 'w') as f:
        f.write("business_id_1, business_id_2, similarity\n")
        for line in final_result.collect():
            f.write(str(line[0]) + "," + str(line[1]) + "," + str(line[2]) + "\n")
    print("Duration: ", time.time() - start)


input_file_name, output_file_name = sys.argv[1], sys.argv[2]
task1(input_file_name, output_file_name)