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


def task1(input_filename, stream_size, num_of_asks, output_filename):
    start_time = time.time()
    bx = BlackBox()
    prev_user = set()
    bit_arr = [0] * 69997
    parameter = create_a_b()

    output_list = []
    for index in range(num_of_asks):
        positive_arr = [0] * stream_size
        stream_user = bx.ask(input_filename, stream_size)
        for i in range(len(stream_user)):
            user_code = int(binascii.hexlify(stream_user[i].encode('utf8')), 16)
            positive_arr[i] = 1
            for t in parameter:
                val = ((t[0] * user_code + t[1]) % 1000003) % 69997
                if bit_arr[val] == 0:
                    positive_arr[i] = 0
                    break

        fp = 0.0
        p = sum(positive_arr)
        for i in range(len(stream_user)):
            if positive_arr[i] == 1 and stream_user[i] not in prev_user:
                fp += 1.0
        print(fp, p)

        false_positive_rate = fp / (stream_size - p + fp)
        output_list.append(str(false_positive_rate) + "\n")
        for user in stream_user:
            prev_user.add(user)
            user_code = int(binascii.hexlify(stream_user[i].encode('utf8')), 16)
            for t in parameter:
                val = ((t[0] * user_code + t[1]) % 1000003) % 69997
                bit_arr[val] = 1

    with open(output_filename, 'w') as f:
        f.write("Time,FPR\n")
        for i in range(len(output_list)):
            f.write(str(i) + "," + output_list[i])

    print("Duration:", time.time() - start_time)


input_filename, stream_size, num_of_asks, output_filename = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
task1(input_filename, stream_size, num_of_asks, output_filename)