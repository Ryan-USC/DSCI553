from pyspark import SparkContext
import json
import sys


def task1(reviews, output_file):
    sc = SparkContext(master="local[*]", appName="task1")
    data = sc.textFile(reviews).map(json.loads)
    rdd = data.map(lambda f: (f['user_id'], f['business_id'], f['date'])).persist()

    # A. The total number of reviews
    total_number_review = rdd.count()

    # B. The number of reviews in 2018
    number_review_2018 = rdd.filter(lambda r: '2018' in r[2]).count()

    # C.  The number of distinct users who wrote reviews
    number_distinct_user = rdd.map(lambda r: r[0]).distinct().count()

    # D.  The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote
    review_users = rdd.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)
    top10_users = review_users.takeOrdered(10, key=lambda k: (-k[1], k[0]))

    # E. The number of distinct businesses that have been reviewed
    review_business = rdd.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b)
    number_distinct_business = review_business.count()

    # F. The top 10 businesses that had the largest numbers of reviews and the number of reviews they had
    top10_business = review_business.takeOrdered(10, key=lambda k: (-k[1], k[0]))

    output = {'n_review': total_number_review, 'n_review_2018': number_review_2018, 'n_user': number_distinct_user,
              'top10_user': top10_users, 'n_business': number_distinct_business, 'top10_business': top10_business}

    with open(output_file, 'w') as file:
        json.dump(output, file)
    sc.stop()


review_filepath = sys.argv[1]
output_filepath = sys.argv[2]

task1(review_filepath, output_filepath)
