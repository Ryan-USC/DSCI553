import sys
import time
import json
from pyspark import SparkContext


def task3(review, business, output_a, output_b):
    sc = SparkContext(master="local[*]", appName="task3")
    review = sc.textFile(review).map(json.loads).map(lambda x: (x['business_id'], x['stars']))
    business = sc.textFile(business).map(json.loads).map(lambda x: (x['business_id'], x['city']))

    data = review.join(business).map(lambda x: (x[1][1], x[1][0])).groupByKey().map(
        lambda x: (x[0], sum(x[1]) / len(x[1])))

    start_m1 = time.time()
    data_a = data.collect()
    data_a.sort(key=lambda x: (-x[1], x[0]))
    for s in data_a[:10]:
        print(s)
    end_m1 = time.time()

    start_m2 = time.time()
    print(data.takeOrdered(10, key=lambda x: (-x[1], x[0])))
    end_m2 = time.time()

    output = {'m1': end_m1 - start_m1, 'm2': end_m2 - start_m2,
              'reason': "The first one returns part of rdd and the second one return the entire rdd, so the first one is faster."}

    with open(output_a, 'w') as file_a:
        file_a.write("city,stars\n")
        for line in data_a:
            file_a.write(line[0]+","+str(line[1])+"\n")

    with open(output_b, 'w') as file_b:
        json.dump(output, file_b)
    sc.stop()


review_filepath = sys.argv[1]
business_filepath = sys.argv[2]
output_filepath_question_a = sys.argv[3]
output_filepath_question_b = sys.argv[4]

task3(review_filepath, business_filepath, output_filepath_question_a, output_filepath_question_b)
