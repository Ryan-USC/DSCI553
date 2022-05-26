import sys
import time
import random
import binascii
from blackbox import BlackBox


def create_a_b():
    a_list = random.sample(range(1, sys.maxsize), 4)
    b_list = random.sample(range(1, sys.maxsize), 4)
    parameter = []
    for k in range(4):
        parameter.append(tuple((a_list[k], b_list[k])))
    return parameter


def myhashs(s):
    result = []
    hash_function_list = create_a_b()
    for f in hash_function_list:
        user_code = int(binascii.hexlify(s.encode('utf8')), 16)
        val = ((f[0] * user_code + f[1]) % 1000003) % 69997
        result.append(val)
    return result


def task2(input_filename, stream_size, num_of_asks, output_filename):
    start_time = time.time()
    bx = BlackBox()

    parameter = create_a_b()
    trailing_0 = [0] * 4
    ground_truth = []
    estimate_list = []

    for index in range(num_of_asks):
        stream_user = bx.ask(input_filename, stream_size)
        users = set()
        for u in stream_user:
            users.add(u)
        for i in range(len(stream_user)):
            user_code = int(binascii.hexlify(stream_user[i].encode('utf8')), 16)
            for k in range(len(parameter)):
                val = ((parameter[k][0] * user_code + parameter[k][1]) % 1000003) % 69997
                zeros = len(bin(val).split("1")[-1])
                trailing_0[k] = max(zeros, trailing_0[k])
        estimate_cnt = 0
        for l in trailing_0:
            estimate_cnt += (2 ** l) / 4
        estimate_list.append(int(estimate_cnt))
        ground_truth.append(len(users))
        trailing_0 = [0] * 4
    sum_of_estimation = sum(estimate_list)
    sum_of_ground_truth = sum(ground_truth)
    print(sum_of_estimation / sum_of_ground_truth)
    with open(output_filename, 'w') as f:
        f.write("Time,Ground Truth,Estimation\n")
        for i in range(len(estimate_list)):
            f.write(str(i) + "," + str(ground_truth[i]) + "," + str(estimate_list[i]) + "\n")

    print("Duration:", time.time() - start_time)


input_filename, stream_size, num_of_asks, output_filename = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
task2(input_filename, stream_size, num_of_asks, output_filename)