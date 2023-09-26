import pandas as pd
import numpy as np
from Extension.record import Record
import matplotlib.pyplot as plt
import os
import psutil
import time
import multiprocessing
import copy

'''adopted from chen_estimate'''
'''the goal for this file is to realize the slicing of record window,
it is rather hard to implement this directly using .txt file,
so firstly write a python program to realize it.'''


def romee_estimate_for_single_value(enviornment, delta_i, n, alpha, window_width1 = -1, window_width2 = -1):
    pid = os.getpid()

    mistake_duration = 0
    next_expected_arrival_time = enviornment[0]
    '''why next_expected_arrival_time here is environment[0]? only for convenience?'''
    record = Record(n)
    wrong_count = 0

    if window_width1 == -1 and window_width2 != -1:
        window_width1, window_width2 = window_width2, window_width1

    if window_width2 == -1:
        '''this means the user is not using two windows, there is only one window'''
        for arrival_time in enviornment:
            record.append(arrival_time)
            current_length = record.get_length()
            current_sum = record.get_sum(window_width1)

            if arrival_time > next_expected_arrival_time:
                mistake_duration += arrival_time - next_expected_arrival_time
                wrong_count += 1

            if window_width1 == -1 or current_length < window_width1:
                '''thie means the user has no requirement on the window width or the current appended records are shorter than the window width, in this case, we use current length to do the calculation.'''
                next_expected_arrival_time = alpha + current_sum / current_length + ((current_length + 1) / 2) * delta_i
            else:
                next_expected_arrival_time = alpha + current_sum / window_width1 + ((window_width1 + 1) / 2) * delta_i

    else:
        '''this is the case that user specifies both the first window width and the second window width'''
        for arrival_time in enviornment:
            record.append(arrival_time)
            current_length = record.get_length()
            current_sum1 = record.get_sum(window_width1)
            current_sum2 = record.get_sum(window_width2)

            if arrival_time > next_expected_arrival_time:
                mistake_duration += arrival_time - next_expected_arrival_time
                wrong_count += 1

            next_expected_arrival_time1 = alpha + current_sum1 / (current_length if (current_length < window_width1) else window_width1) + (((current_length if (current_length < window_width1) else window_width1) + 1) / 2) * delta_i

            next_expected_arrival_time2 = alpha + current_sum2 / (current_length if (current_length < window_width2) else window_width2) + (((current_length if (current_length < window_width2) else window_width2) + 1) / 2) * delta_i

            next_expected_arrival_time = max(next_expected_arrival_time1, next_expected_arrival_time2)

    # for arrival_time in enviornment:
    #     record.append(arrival_time)
    #     current_length = record.get_length()
    #     current_sum = record.get_sum()

    #     if arrival_time > next_expected_arrival_time:
    #         mistake_duration += arrival_time - next_expected_arrival_time
    #         wrong_count += 1

    #     next_expected_arrival_time = alpha + current_sum / current_length + ((current_length + 1) / 2) * delta_i

    detection_time = next_expected_arrival_time - enviornment[-1]
    if detection_time < 0:
        detection_time = 0
    pa = (len(enviornment) - wrong_count) / len(enviornment)
    cpu_time = psutil.Process(pid).cpu_times().system
    memory = psutil.Process(pid).memory_info().rss / 1024 / 1024

    # q.put((mistake_duration, detection_time, pa, cpu_time, memory))
    return mistake_duration, detection_time, pa, cpu_time, memory

'''
def chen_estimate_for_alpha_array(enviornment, delta_i, n, alpha_list):
    mistake_duration = np.zeros(len(alpha_list), dtype=float)
    next_expected_arrival_time = np.array([float('inf') for i in range(len(alpha_list))])
    record = Record(n)
    for arrival_time in enviornment:
        record.append(arrival_time)
        current_length = record.get_length()
        current_sum = record.get_sum()

        duration = -next_expected_arrival_time + arrival_time
        duration = np.maximum(duration, 0)
        mistake_duration += duration

        next_expected_arrival_time = alpha_list + current_sum / current_length + (
                (current_length + 1) / 2) * delta_i

    return mistake_duration


def chen_estimate_for_n_array(enviornment, delta_i, n_list, alpha):
    mistake_duration = np.zeros(len(n_list), dtype=float)
    next_expected_arrival_time = np.array([float('inf') for i in range(len(n_list))])
    record_list = [Record(i) for i in n_list]
    current_length = np.zeros(len(n_list))
    current_sum = np.zeros(len(n_list), dtype=float)
    for arrival_time in enviornment:
        for inx, record in enumerate(record_list):
            record.append(arrival_time)
            current_length[inx] = record.get_length()
            current_sum[inx] = record.get_sum()

        duration = -next_expected_arrival_time + arrival_time
        duration = np.maximum(duration, 0)
        mistake_duration += duration

        next_expected_arrival_time = alpha + current_sum / current_length + (
                (current_length + 1) / 2) * delta_i

    return mistake_duration


def chen_estimate(enviornment, delta_i, n_list, alpha_list):
    if type(n_list) != np.ndarray and type(n_list) != int:
        raise TypeError('The data type of n can only be numpy array or int')
    if type(alpha_list) != np.ndarray and type(alpha_list) != int:
        raise TypeError('The data type of alpha can only be numpy array or int')

    if type(n_list) == np.ndarray and type(alpha_list) == np.ndarray:
        raise TypeError('The data type of n and alpha cannot be both array')
    elif type(alpha_list) == np.ndarray:
        mistake_duration = chen_estimate_for_alpha_array(enviornment, delta_i, n_list, alpha_list)
        return mistake_duration
    elif type(n_list) == np.ndarray:
        mistake_duration = chen_estimate_for_n_array(enviornment, delta_i, n_list, alpha_list)
        return mistake_duration
    else:
        mistake_duration = chen_estimate_for_single_value(enviornment, delta_i, n_list, alpha_list)
        return mistake_duration
'''

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)

    delta_i = 100000000.0
    n = 1000
    alpha = 100000
    node_list = [0, 1, 3, 5, 6, 7, 8, 9]
    window_width1 = 3
    window_width2 = 7
    pool = multiprocessing.Pool(processes=56)
    results = []
    for i in node_list:
        receive_from_node_list = copy.deepcopy(node_list)
        receive_from_node_list.remove(i)
        for j in receive_from_node_list:
            filename = os.path.join(dirname, "data", f"Node{i}", "trace.csv")
            df = pd.read_csv(filename)
            df = df[df.site == j]
            arrival_time_array = np.array(df.timestamp_receive)
            results.append(pool.apply_async(romee_estimate_for_single_value, (arrival_time_array, delta_i, n, alpha, window_width1, window_width2)))

    pool.close()
    pool.join()
    mistake_duration_list = []
    detection_time_list = []
    pa_list = []
    cpu_time_list = []
    memory_list = []
    for res in results:
        mistake_duration_list.append(res.get()[0] / 1000000)
        detection_time_list.append(res.get()[1] / 1000000)
        pa_list.append(res.get()[2])
        cpu_time_list.append(res.get()[3])
        memory_list.append(res.get()[4])

    mistake_duration_array = np.array(mistake_duration_list)
    detection_time_array = np.array(detection_time_list)
    pa_array = np.array(pa_list)
    cpu_time_array = np.array(cpu_time_list)
    memory_array = np.array(memory_list)

    print(f"average mistake duration: {np.mean(mistake_duration_array):.2f} ms")
    print(f"average detection time: {np.mean(detection_time_array):.2f} ms")
    print(f"average pa: {np.mean(pa_array):.2%}")
    print(f"average cpu time: {np.mean(cpu_time_array):.2f} s")
    print(f"average memory: {np.mean(memory_array):.2f} MB")
    print(f"std detection time: {np.std(detection_time_array):.2f} ms")
    print(f"std pa: {np.std(pa_array):.2%}")

    # df = pd.read_csv(r'.\data\Node0\trace.csv')
    # df = df[df.site == 8]
    # arrival_time_array = np.array(df.timestamp_receive)
    #
    # delta_i = 100000000.0
    # # # n_list = np.array([i for i in range(1, 101)])
    # n = 1000
    # # alpha_list = np.array([0, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000], dtype=float)
    # alpha = 100000
    #
    # mistake_duration, detection_time, pa, cpu_time, memory = chen_estimate_for_single_value(arrival_time_array, delta_i, n, alpha)
    #
    # print(f"{mistake_duration / 1000000:.2f} ms")
    # print(f"{detection_time / 1000000:.2f} ms")
    # print(f"{pa:.2%}")
    # print(f"{cpu_time:.2f} s")
    # print(f"{memory:.2f} MB")
    # #
    # # plt.plot(alpha_list, mistake_duration)
    # # plt.show()