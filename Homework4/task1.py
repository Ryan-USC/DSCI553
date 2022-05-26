from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import itertools
import graphframes
import os
import sys
import time

os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"


def task1(filter_threshold, input_path, output_path):
    start = time.time()

    conf = SparkConf().setAppName("DSCI553").setMaster('local[*]')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    sqlContext = SQLContext(sc)

    rdd = sc.textFile(input_path)
    header = rdd.first()
    ub_rdd = rdd.filter(lambda x: x != header).map(lambda x: x.split(',')).cache()

    uid_rdd = ub_rdd.map(lambda x: x[0])
    distinct_user = uid_rdd.distinct().collect()
    user_map = ub_rdd.groupByKey().collectAsMap()

    # find connected nodes set
    vertex_set = set()
    node_pair = set()
    for pair in itertools.combinations(distinct_user, 2):
        if len(set(user_map[pair[0]]).intersection(set(user_map[pair[1]]))) >= filter_threshold:
            node_pair.add((pair[0], pair[1]))
            vertex_set.add(pair[0])
            vertex_set.add(pair[1])

    # 4.1 Graph Construction
    vertices = sqlContext.createDataFrame([(uid,) for uid in vertex_set]).toDF("id")
    edges = sqlContext.createDataFrame(list(node_pair)).toDF("src", "dst")
    graph = graphframes.GraphFrame(vertices, edges)


    # 4.2.2 find communities and output result
    res = graph.labelPropagation(maxIter=5)
    community_rdd = res.rdd.map(lambda x: (x[1], x[0])).groupByKey(). \
        map(lambda label: sorted(list(label[1]))).sortBy(lambda l: (len(l), l)).collect()

    with open(output_path, 'w') as f:
        for id in community_rdd:
            f.write(str(id)[1:-1] + "\n")

    print("Duration: ", time.time() - start)


filter_threshold, input_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
task1(filter_threshold, input_path, output_path)