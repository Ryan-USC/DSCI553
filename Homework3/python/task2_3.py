import os
import sys
import time
import math
import json
import numpy as np
import xgboost as xgb
from pyspark import SparkContext, SparkConf


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


def get_noise_level(attributes):
    if attributes:
        if "NoiseLevel" in attributes.keys():
            if attributes["NoiseLevel"] == 'quiet':
                return 1
            elif attributes["NoiseLevel"] == 'average':
                return 2
            elif attributes["NoiseLevel"] == 'loud':
                return 3
            elif attributes["NoiseLevel"] == 'very_loud':
                return 4
    return 2


def task2_3(folder_path, test_file_name, output_file_name):
    def rate_no_business(user):
        if user not in user_rate.keys():
            return "2.5"
        else:
            return str(user_avg.get(user))

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
    train_file_name = os.path.join(folder_path, 'yelp_train.csv')
    user_json_path = os.path.join(folder_path, 'user.json')
    business_json_path = os.path.join(folder_path, 'business.json')

    conf = SparkConf().setAppName("DSCI553").setMaster('local[*]')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    trainRdd = sc.textFile(train_file_name)
    trainHeader = trainRdd.first()
    trainData = trainRdd.filter(lambda x: x != trainHeader).map(lambda x: x.split(','))

    testRdd = sc.textFile(test_file_name)
    testHeader = testRdd.first()
    testData = testRdd.filter(lambda x: x != testHeader).map(lambda x: x.split(','))
    user_rate = trainData.map(lambda x: (x[0], (x[1], float(x[2])))). \
        groupByKey(). \
        mapValues(dict). \
        collectAsMap()
    business_rate = trainData.map(lambda x: ((x[1]), ((x[0]), float(x[2])))). \
        groupByKey(). \
        mapValues(dict). \
        collectAsMap()
    user_avg = trainData.map(lambda x: (x[0], float(x[2]))). \
        groupByKey(). \
        mapValues(lambda x: sum(x) / len(x)). \
        collectAsMap()
    business_avg = trainData.map(lambda x: (x[1], float(x[2]))). \
        groupByKey(). \
        mapValues(lambda x: sum(x) / len(x)). \
        collectAsMap()
    prediction = testData.sortBy(lambda x: (x[0], x[1])).map(item_based_collaborative_filtering)

    user_json_rdd = sc.textFile(user_json_path). \
        map(json.loads). \
        map(lambda x: ((x["user_id"]), (x["review_count"], x["useful"], x["average_stars"]))). \
        persist()
    user_json_dict = user_json_rdd.collectAsMap()
    business_json_rdd = sc.textFile(business_json_path) \
        .map(json.loads) \
        .map(lambda x: ((x['business_id']), (x['stars'], x['review_count'], get_noise_level(x['attributes'])))) \
        .persist()
    business_json_dict = business_json_rdd.collectAsMap()

    x_train = np.array(
        trainData.map(lambda row: np.array(user_json_dict.get(row[0]) + business_json_dict.get(row[1]))).collect())
    y_train = np.array(trainData.map(lambda row: float(row[2])).collect())
    xgbr = xgb.XGBRegressor(seed=20)
    xgbr.fit(x_train, y_train)
    x_test = np.array(
        testData.map(lambda row: np.array(user_json_dict.get(row[0]) + business_json_dict.get(row[1]))).collect())
    y_pred = xgbr.predict(x_test)
    id_pair = testData.map(lambda row: (row[0], row[1])).collect()
    pred_map = {}
    for i in range(len(y_pred)):
        pred_map[id_pair[i]] = y_pred[i]

    result = prediction.map(lambda x: (((x[0]), (x[1])), 0.1 * float(x[2]) + 0.9 * pred_map.get((x[0], x[1]))))
    resultList = result.collect()
    with open(output_file_name, 'w') as f:
        f.write("user_id, business_id, prediction\n")
        for i in range(len(resultList)):
            f.write(str(resultList[i][0]) + "," + str(resultList[i][1]) + "," + str(resultList[i][2]) + "\n")
    print(time.time() - start)


folder_path, test_file_name, output_file_name = sys.argv[1], sys.argv[2], sys.argv[3]
task2_3(folder_path, test_file_name, output_file_name)
# RMSE 0.983825786103625
