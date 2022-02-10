import sys
import json
import time
from pyspark import SparkContext
import findspark


if __name__ == '__main__':
    findspark.init()
    review_filepath, output_filepath, n_partition = sys.argv[1], sys.argv[2], sys.argv[3]

    sc = SparkContext(master="local[*]", appName="task2")
    rdd = sc.textFile(review_filepath).map(json.loads)

    def getPartition(rdd):
        partition_n = rdd.getNumPartitions()
        n_items = rdd.glom().map(len).collect()
        rdd.reduceByKey(lambda a, b: a + b).takeOrdered(10, key=lambda f: (-f[1], f[0]))
        return partition_n, n_items

    default_start = time.time()
    default_rdd = rdd.map(lambda x: (x['business_id'], 1)).persist()
    default_n_partition, default_n_items = getPartition(default_rdd)
    default_end = time.time()
    default_time = default_end - default_start

    custom_start = time.time()
    customized_rdd = rdd.map(lambda x: (x['business_id'], 1)).partitionBy(int(n_partition),
                                                                          lambda x: ord(x[0][0]) % int(
                                                                              n_partition))
    custom_n_partition, custom_n_items = getPartition(customized_rdd)
    custom_end = time.time()
    custom_time = custom_end - custom_start

    output = {'default': {'n_partition': default_n_partition,
                          'n_items': default_n_items,
                          'exe_time': default_time},
              'customized': {'n_partition': custom_n_partition,
                             'n_items': custom_n_items,
                             'exe_time': custom_time}}

    with open(output_filepath, 'w') as file:
        json.dump(output, file)
    sc.stop()
