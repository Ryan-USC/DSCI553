from pyspark import SparkConf, SparkContext
import time
import math
import csv
import sys


def pearson(neighbor, user_set, user_dict, business_rating_avg, neighbor_rating_avg, business_rate):
    user_neighbor_set = set(business_rate.get(neighbor).keys())
    user_neighbor_dict = dict(business_rate.get(neighbor))
    commons = user_set.intersection(user_neighbor_set)
    if len(commons) == 0:
        return float(business_rating_avg / neighbor_rating_avg)
    numerator = 0
    denominator_business = 0
    denominator_neighbor = 0
    for user in commons:
        normalized_business = user_dict.get(user) - business_rating_avg
        normalized_neighbor = user_neighbor_dict.get(user) - neighbor_rating_avg
        numerator += normalized_business * normalized_neighbor
        denominator_business += normalized_business * normalized_business
        denominator_neighbor += normalized_neighbor * normalized_neighbor
    denominator = math.sqrt(denominator_business * denominator_neighbor)
    if denominator == 0:
        if numerator == 0:
            res = 1
        else:
            res = -1
    else:
        res = numerator / denominator
    return res


def task2_1(train_file_name, test_file_name, output_file_name):
    def rate_no_business(user):
        if user not in user_rate.keys():
            return "2.5"
        else:
            return str(user_avg.get(user))

    def rating_prediction(weight):
        n = min(len(weight), 30)
        weight.sort(key=lambda x: x[0], reverse=True)
        weight = weight[:n]
        sum_similarity = 0
        sum_weight = 0
        for i in range(n):
            similarity = weight[i][0]
            rate = weight[i][1]
            sum_similarity += abs(similarity)
            sum_weight += similarity * rate
        return sum_weight / sum_similarity

    def item_based_collaborative_filtering(line):
        user = line[0]
        business = line[1]
        if business not in business_rate.keys():
            rate = rate_no_business(user)
            return user, business, rate
        else:
            business_rating_avg = business_avg.get(business)
            if user not in user_rate.keys():
                return user, business, str(business_rating_avg)
            else:
                businesses = user_rate.get(user).keys()
                user_dict = dict(business_rate.get(business))
                user_set = set(business_rate.get(business).keys())
                weight = []
                for neighbor in businesses:
                    if neighbor == business:
                        continue
                    cur_neighbor_rate = business_rate.get(neighbor).get(user)
                    neighbor_rating_avg = business_avg.get(neighbor)
                    pearson_coef = pearson(neighbor, user_set, user_dict, business_rating_avg, neighbor_rating_avg,
                                           business_rate)
                    if pearson_coef > 1:
                        pearson_coef = 1 / pearson_coef
                    if pearson_coef > 0:
                        weight.append((pearson_coef, cur_neighbor_rate))
                return user, business, rating_prediction(weight)

    start = time.time()
    conf = SparkConf().setAppName("DSCI553").setMaster('local[*]')
    sc = SparkContext(conf=conf)

    train_rdd = sc.textFile(train_file_name)
    train_header = train_rdd.first()
    train_data = train_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(','))
    test_rdd = sc.textFile(test_file_name)
    test_header = test_rdd.first()
    test_data = test_rdd.filter(lambda x: x != test_header).map(lambda x: x.split(','))

    user_rate = train_data.map(lambda x: (x[0], (x[1], float(x[2])))). \
        groupByKey(). \
        mapValues(dict). \
        collectAsMap()
    business_rate = train_data.map(lambda x: ((x[1]), ((x[0]), float(x[2])))). \
        groupByKey(). \
        mapValues(dict). \
        collectAsMap()
    user_avg = train_data.map(lambda x: (x[0], float(x[2]))). \
        groupByKey(). \
        mapValues(lambda x: sum(x) / len(x)). \
        collectAsMap()
    business_avg = train_data.map(lambda x: (x[1], float(x[2]))). \
        groupByKey(). \
        mapValues(lambda x: sum(x) / len(x)). \
        collectAsMap()

    prediction = test_data.sortBy(lambda x: (x[0], x[1])).map(item_based_collaborative_filtering).collect()

    with open(output_file_name, 'w') as f:
        f.write("user_id, business_id, prediction\n")
        for i in range(len(prediction)):
            f.write(str(prediction[i][0]) + "," + str(prediction[i][1]) + "," + str(prediction[i][2]) + "\n")

    print(time.time() - start)


train_file_name, test_file_name, output_file_name = sys.argv[1], sys.argv[2], sys.argv[3]
task2_1(train_file_name, test_file_name, output_file_name)
