# Method Description:
# In order to improve the recommender system, what I choose is using model-based CF which has better performance
# To improve the performance, I added another two attribute which are "fan" in user.json and "wifi" from business.json
# And also, I change the parameter of default value of noise level which also improve a little of the RMSE.
#
#Error Distribution:
# >=0 and <1: 101971
# >=1 and <2: 33037
# >=2 and <3: 6195
# >=3 and <4: 840
# >=4: 1
#
# RMSE:
# 0.9819253043558718
#
# Esecution Time:
# 47.41270709037781s
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from pyspark import SparkContext, SparkConf


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
    return 3


def have_wifi(attributes):
    if attributes:
        if "WiFi" in attributes.keys():
            if attributes["WiFi"] == 'free':
                return 2
            elif attributes["WiFi"] == 'no':
                return 0
        else:
            return -1
    return -1


def get_data(data, user_map, business_map, default_user, default_business):
    output = {}
    user_review_count = []
    user_userful = []
    user_stars = []
    user_fans = []
    business_review_count = []
    business_stars = []
    business_noise = []
    business_wifi = []
    for user in data['user_id']:
        if user in user_map.keys():
            user_review_count.append(user_map.get(user)[0])
            user_userful.append(user_map.get(user)[1])
            user_stars.append(user_map.get(user)[2])
            user_fans.append(user_map.get(user)[3])
        else:
            user_review_count.append(default_user['review_count'])
            user_userful.append(default_user['userful'])
            user_stars.append(default_user['stars'])
            user_fans.append(default_user['fans'])
    for business in data['business_id']:
        if business in business_map.keys():
            business_review_count.append(business_map.get(business)[1])
            business_stars.append(business_map.get(business)[0])
            business_noise.append(business_map.get(business)[2])
            business_wifi.append(business_map.get(business)[3])
        else:
            business_review_count.append(default_business['review_count'])
            business_stars.append(default_business['stars'])
            business_noise.append(default_business['noise'])
            business_wifi.append(default_business['wifi'])

    output['user_review_count'] = user_review_count
    output['useful'] = user_userful
    output['user_stars'] = user_stars
    output['fans'] = user_fans
    output['business_review_count'] = business_review_count
    output['business_stars'] = business_stars
    output['noise'] = business_noise
    output['wifi'] = business_wifi

    return output


def competition(folder_path,  test_file_name,  output_file_name):

    start_time = time.time()
    # folder_path = 'data/'
    # test_file_name = 'yelp_val.csv'
    # output_file_name = 'task2_2.csv'
    train_file = os.path.join(folder_path, 'yelp_train.csv')
    user_json_path = os.path.join(folder_path, 'user.json')
    business_json_path = os.path.join(folder_path, 'business.json')

    conf = SparkConf().setAppName("DSCI553").setMaster('local[*]')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file_name)

    user_json_rdd = sc.textFile(user_json_path). \
            map(json.loads). \
            map(lambda x: ((x["user_id"]), (x["review_count"], x["useful"], x["average_stars"], x["fans"]))). \
            persist()
    user_json_dict = user_json_rdd.collectAsMap()

    business_json_rdd = sc.textFile(business_json_path) \
            .map(json.loads) \
            .map(lambda x: ((x['business_id']), (x['stars'], x['review_count'], get_noise_level(x['attributes']), have_wifi(x['attributes'])))) \
            .persist()
    business_json_dict = business_json_rdd.collectAsMap()

    default_user = {'review_count': user_json_rdd.map(lambda x: x[1][0]).mean(),
                        'userful': 0,
                        'stars': user_json_rdd.map(lambda x: x[1][2]).mean(),
                    "fans": user_json_rdd.map(lambda x: x[1][3]).mean()}
    default_business = {'review_count': business_json_rdd.map(lambda x: x[1][1]).mean(),
                            'stars': business_json_rdd.map(lambda x: x[1][0]).mean(),
                            'noise': business_json_rdd.map(lambda x: x[1][2]).mean(),
                        "wifi": business_json_rdd.map(lambda x: x[1][3]).mean()}
    training_data = pd.DataFrame.from_dict(
            get_data(train_data, user_json_dict, business_json_dict, default_user, default_business))
    trainX = np.array(training_data)
    trainY = train_data.stars.values

    xgbr = xgb.XGBRegressor(seed=20)
    xgbr.fit(trainX, trainY)

    testing_data = pd.DataFrame.from_dict(
            get_data(test_data, user_json_dict, business_json_dict, default_user, default_business))
    testX = np.array(testing_data)
    testY = test_data.stars.values
    prediction = xgbr.predict(testX)

    result = pd.DataFrame()
    result["user_id"] = test_data.user_id.values
    result["business_id"] = test_data.business_id.values
    result["prediction"] = prediction
    result.to_csv(output_file_name, header=['user_id', 'business_id', 'prediction'], index=False, sep=',', mode='w')

    print("Duration: ", time.time() - start_time)


folder_path,  test_file_name,  output_file_name = sys.argv[1], sys.argv[2], sys.argv[3]
competition(folder_path,  test_file_name,  output_file_name)

