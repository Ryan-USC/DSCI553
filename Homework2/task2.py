from pyspark import SparkContext, SparkConf
import itertools
import math
import time
import sys


def data_process(input_file, intermediate):
    file = open(input_file, "r", encoding='utf-8')
    lines = []
    for line in file.readlines()[1:]:
        line = line.strip().split(',')
        year = line[0][:-4] + line[0][-2:]
        customer = line[1].lstrip('0')
        combined = year + "-" + customer
        cur = [combined.replace('"', ''), line[5].replace('"', '').lstrip("0")]
        lines.append(cur)
    with open(intermediate, 'w') as f:
        f.write("DATE-CUSTOMER_ID,PRODUCT_ID\n")
        for l in lines:
            f.write(",".join(l) + "\n")
    return


def get_candidate(chunk, idx, candidate_prev):
    output = {}
    for basket in chunk:
        common = sorted(list(set(basket).intersection(candidate_prev)))
        for item in itertools.combinations(common, idx):
            output[item] = output.get(item, 0) + 1
    return output


def get_singleton(chunk):
    items = {}
    for basket in chunk:
        for item in basket:
            item = (item,)
            items[item] = items.get(item, 0) + 1
    return items


def filter_by_support(items, support_chunk):
    output = []
    for item, count in items.items():
        if count >= support_chunk:
            output.append(item)
    return output


def apriori(partition, support, basket_count):
    output = []
    chunk = list(partition)
    support_chunk = math.ceil(len(chunk) * support / basket_count)

    items = get_singleton(chunk)
    candidates = filter_by_support(items, support_chunk)

    if len(candidates) > 0:
        output.append(candidates)
    else:
        return output
    candidates = set(x[0] for x in candidates)
    idx = 2
    while True:
        items = get_candidate(chunk, idx, candidates)
        candidates = filter_by_support(items, support_chunk)

        if len(candidates) > 0:
            output.append(candidates)
        else:
            break
        candidates = set([item for basket in candidates for item in basket])
        idx += 1
    return output


def get_frequent(partition, candidate_rdd):
    frequency = {}
    for item in partition:
        for c in candidate_rdd:
            if set(item).issuperset(set(c)):
                frequency[c] = frequency.get(c, 0) + 1
    output = []
    for key, value in frequency.items():
        output.append((key, value))
    return output


def get_output_string(rdd):
    output = ""
    candidate_list = list(rdd)
    length = 1
    for item in candidate_list:
        if len(item) != length:
            length += 1
            output = output[:-1]
            output += "\n\n"
        if len(item) == 1:
            output += str(item).replace(",", "") + ","
        else:
            output += str(item) + ","
    output = output[:-1]
    return output


def write_to_file(output_file, candidate_res, frequent_res):
    with open(output_file, 'w') as file:
        file.write("Candidates:\n" + candidate_res + "\n\nFrequent Itemsets:\n" + frequent_res)


def task2(filter_threshold, support, input_file, output_file):
    intermediate = "customer_product.csv"
    data_process(input_file, intermediate)

    conf = SparkConf().setAppName("DSCI553-HW2").setMaster('local[*]')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    start = time.time()
    rdd = sc.textFile(intermediate)
    header = rdd.first()
    data = rdd.filter(lambda r: r != header).map(lambda row: row.strip().split(","))

    baskets = data.map(lambda row: (str(row[0]), str(row[1]))).groupByKey().mapValues(list). \
        filter(lambda row: len(row[1]) > filter_threshold).map(lambda row: sorted(list(set(row[1]))))
    baskets_count = baskets.count()

    candidate_rdd = baskets.mapPartitions(lambda partition: apriori(partition, support, baskets_count)).flatMap(
        lambda basket: [x for x in basket])
    candidate_rdd = candidate_rdd.distinct().sortBy(lambda x: (len(x), x)).collect()
    candidate_res = get_output_string(candidate_rdd)

    frequent_rdd = baskets.mapPartitions(lambda partition: get_frequent(partition, candidate_rdd))
    frequent_rdd = frequent_rdd.reduceByKey(lambda a, b: a + b).filter(lambda x: x[1] >= support).map(lambda x: x[0])
    frequent_rdd = frequent_rdd.sortBy(lambda x: (len(x), x)).collect()
    frequent_res = get_output_string(frequent_rdd)

    write_to_file(output_file, candidate_res, frequent_res)
    end = time.time()
    print("Duration:", end - start)


filter_threshold, support, input_file, output_file = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]

task2(filter_threshold, support, input_file, output_file)
