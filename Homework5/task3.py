import sys
import time
import random
from blackbox import BlackBox


def task3(input_filename, stream_size, num_of_asks, output_filename):
    start_time = time.time()
    random.seed(553)
    bx = BlackBox()

    sample = []
    n = 100
    sequence = []
    for index in range(num_of_asks):
        stream_user = bx.ask(input_filename, stream_size)
        if index == 0:
            for user in stream_user:
                sample.append(user)
        else:
            for user in stream_user:
                n += 1
                prob = random.random()
                if prob > 100 / n:
                    sample[random.randint(0, 99)] = user
        sequence.append(
            str(n) + "," + sample[0] + "," + sample[20] + "," + sample[40] + "," + sample[60] + "," + sample[80] + "\n")

    with open(output_filename, 'w') as f:
        f.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")
        for i in range(len(sequence)):
            f.write(sequence[i])
    print("Duration:", time.time() - start_time)


input_filename, stream_size, num_of_asks, output_filename = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
task3(input_filename, stream_size, num_of_asks, output_filename)
