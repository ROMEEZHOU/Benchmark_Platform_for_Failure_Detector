import pandas as pd
import numpy as np
from Extension.record import Record
import matplotlib.pyplot as plt
import os
import psutil
import time
import multiprocessing
import copy


def romee_shitty_for_single_value(enviornment, n, alpha):
    pid = os.getpid()

    mistake_duration = 0
    next_expected_arrival_time = enviornment[0]
    record = Record(n)
    wrong_count = 0
    for arrival_time in enviornment:
        record.append(arrival_time)
        current_length = record.get_length()
        current_diff = record.get_difference()

        if arrival_time > next_expected_arrival_time:
            mistake_duration += arrival_time - next_expected_arrival_time
            wrong_count += 1

        next_expected_arrival_time = arrival_time + sum(current_diff)/current_length + alpha

    detection_time = next_expected_arrival_time - enviornment[-1]
    if detection_time < 0:
        detection_time = 0
    pa = (len(enviornment) - wrong_count) / len(enviornment)
    cpu_time = psutil.Process(pid).cpu_times().system
    memory = psutil.Process(pid).memory_info().rss / 1024 / 1024

    # q.put((mistake_duration, detection_time, pa, cpu_time, memory))
    return mistake_duration, detection_time, pa, cpu_time, memory


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    #record_file = open('chen_result.txt','w')
    delta_i = 100000000.0
    n_list = np.arange(20,220,20)
    a_list = np.arange(1000000,7000000,1000000)
    print(len(n_list))
    pa_result = []
    dt_average_result = []
    dt_std_result = []
    alpha = 4000000
    #here need to change to iterate through different alpha
    n = 20
    for n in n_list:
        print(alpha)
        # n = 1000
        #alpha = 7000000
        node_list = [0, 1, 3]
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
                results.append(pool.apply_async(romee_shitty_for_single_value, (arrival_time_array, n, alpha,)))

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

        current_pa = np.mean(pa_array)
        dt_average = np.mean(detection_time_array)
        dt_std = np.std(pa_array)
        pa_result.append(current_pa)
        dt_average_result.append(dt_average)
        dt_std_result.append(dt_std)
        
        # record_file.write('chen '+ 'n= '+str(n)+' alpha = '+str(alpha) + '\n')
        # record_file.write('pa:'+str(current_pa)+'\n')
        # record_file.write('standard pa: '+str(np.std(pa_array))+'\n')
        # record_file.write('average detection time: '+str(dt_average)+'\n')
        # record_file.write('detection time standard deviation: '+str(dt_std)+'\n')
        # record_file.write('average cpu time: '+str(np.mean(cpu_time_array))+'\n')
        # record_file.write('\n')

    # trace_mistake = pd.DataFrame({"a": a_list, "d_a": dt_average_result, "d_s": dt_std_result, "pa": pa_result})
    # plot = trace_mistake.plot(x = 'a', y = 'd_a', title = 'Average Detection to alpha')
    # plot2 = trace_mistake.plot(x = 'a', y = 'd_s', title = 'Detection std to alpha')
    # plt.yticks(np.arange(0.5, 1.05, 0.05))
    # plt.plot(a_list, pa_result, label = 'pa to alpha')
    # plt.plot(a_list, dt_average_result, label = 'detection time to alpha')
    # plt.show()

    #record_file.close()

    # Plot the F-x graph
    fig, axl = plt.subplots()
    axl.plot(n_list, pa_result)
    axl.set_xlabel("n")
    axl.set_ylabel("pa", color="tab:blue")
    axl.tick_params(axis="y")

    # Also plot the velocities
    axr = axl.twinx()
    axr.plot(n_list, dt_average_result, color="tab:orange")
    axr.set_ylabel("detection time", color="tab:orange")
    axr.tick_params(axis="y")

    plt.show()

    # print(f"average mistake duration: {np.mean(mistake_duration_array):.2f} ms")
    # print(f"average detection time: {np.mean(detection_time_array):.2f} ms")
    # print(f"average pa: {np.mean(pa_array):.2%}")
    # print(f"average cpu time: {np.mean(cpu_time_array):.2f} s")
    # print(f"average memory: {np.mean(memory_array):.2f} MB")
    # print(f"std detection time: {np.std(detection_time_array):.2f} ms")
    # print(f"std pa: {np.std(pa_array):.2%}")

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
