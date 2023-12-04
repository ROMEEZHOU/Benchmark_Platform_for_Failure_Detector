import pandas as pd
import numpy as np
import scipy.stats as st
import os
from Extension.record import Record
import random
import matplotlib.pyplot as plt

#the standard message transmission time is 1 sec
def naive_trace_generator(length=20,threshold_lower=-1, threshold_upper=1):
    result = []
    for i in range(length):
        current = 1 + random.uniform(threshold_lower,threshold_upper)
        if result == []:
            result.append(current)
        else:
            result.append(result[-1] + current)
    return result

def chen_single(environment, params={'delta_i':1.0, 'n':20, 'alpha':1}):
    #returns the suspicion list that suspector generates
    #also returns the quality of local fd?
    delta_i = params['delta_i']
    n = params['n']
    alpha = params['alpha']
    heartbeat_number = 0

    next_expected_arrival_time = environment[0]
    last_arrival_time = environment[0]
    record = Record(n)
    suspicion = []
    #wrong_count = 0
    for arrival_time in environment:
        heartbeat_number += 1
        record.append(arrival_time)
        current_length = record.get_length()
        current_sum = record.get_sum()

        if arrival_time > next_expected_arrival_time:
            #mistake_duration += arrival_time - next_expected_arrival_time
            #wrong_count += 1
            suspicion.append((heartbeat_number, next_expected_arrival_time,arrival_time, last_arrival_time))

        last_arrival_time = arrival_time
        next_expected_arrival_time = alpha + current_sum / current_length + ((current_length + 1) / 2) * delta_i

    # detection_time = next_expected_arrival_time - enviornment[-1]
    # if detection_time < 0:
    #     detection_time = 0
    # pa = (len(enviornment) - wrong_count) / len(enviornment)
    # cpu_time = psutil.Process(pid).cpu_times().system
    # memory = psutil.Process(pid).memory_info().rss / 1024 / 1024

    return suspicion

def bertier_single(environment, params={'delta_i':1.0, 'n':100, 'delay':0, 'var':0, 'gamma':0.01, 'beta':1, 'phi':5.5}):

    delta_i = params['delta_i']
    n = params['n']
    delay = params['delay']
    var = params['var']
    gamma = params['gamma']
    beta = params['beta']
    phi = params['phi']
    heartbeat_number = 0

    #mistake_duration = 0
    next_expected_arrival_time = environment[0]
    last_arrival_time = environment[0]
    record = Record(n)
    suspicion = []
    #wrong_count = 0
    for arrival_time in environment:
        heartbeat_number += 1
        record.append(arrival_time)
        current_length = record.get_length()
        current_sum = record.get_sum()

        if arrival_time > next_expected_arrival_time:
            # mistake_duration += arrival_time - next_expected_arrival_time
            # wrong_count += 1
            suspicion.append((heartbeat_number, next_expected_arrival_time, arrival_time, last_arrival_time))

        # calculating the value of alpha
        error = arrival_time - next_expected_arrival_time - delay
        delay = delay + gamma * error
        var = var + gamma * (np.abs(error) - var)
        alpha = beta * delay + phi * var

        # calculating the next expected arrival time
        next_expected_arrival_time = alpha + current_sum / current_length + ((current_length + 1) / 2) * delta_i
        last_arrival_time = arrival_time

    return suspicion

def accrual_single(environment, params={'delta_i':1.0, 'n':1000, 'phi':10}):
    delta_i = params['delta_i']
    n = params['n']
    phi = params['phi']
    heartbeat_number = 0

    #mistake_duration = 0
    next_expected_arrival_time = environment[0]
    last_arrival_time = environment[0]
    record = Record(n)
    #wrong_count = 0
    suspicion = []

    for arrival_time in environment:
        heartbeat_number += 1
        record.append(arrival_time)
        difference = record.get_difference()

        if arrival_time > next_expected_arrival_time:
            # mistake_duration += arrival_time - next_expected_arrival_time
            # wrong_count += 1
            suspicion.append((heartbeat_number, next_expected_arrival_time,arrival_time, last_arrival_time))

        # Then start to calculate the expected arrival time
        if len(difference) == 0:
            # Means there are only one arrival time in the record
            next_expected_arrival_time = arrival_time + delta_i
            last_arrival_time = arrival_time
        elif len(difference) == 1:
            # Means there are only two arrival time in the record
            next_expected_arrival_time = arrival_time + difference[0]
            last_arrival_time = arrival_time
        else:
            mean = np.mean(difference)
            scale = np.std(difference)
            expected_interval = st.norm.ppf(1 - np.power(0.1, phi), loc=mean, scale=scale)
            next_expected_arrival_time = arrival_time + expected_interval
            last_arrival_time = arrival_time
    
    return suspicion

def romee_shitty_single(environment, params ={'delta_i':1.0, 'n':200, 'alpha':1}):
    delta_i = params['delta_i']
    n = params['n']
    alpha = params['alpha']
    heartbeat_number = 0

    next_expected_arrival_time = environment[0]
    last_arrival_time = environment[0]
    record = Record(n)
    suspicion = []
    #wrong_count = 0
    for arrival_time in environment:
        heartbeat_number += 1
        record.append(arrival_time)
        current_length = record.get_length()
        current_diff = record.get_difference()

        if arrival_time > next_expected_arrival_time:
            #mistake_duration += arrival_time - next_expected_arrival_time
            #wrong_count += 1
            suspicion.append((heartbeat_number, next_expected_arrival_time,arrival_time, last_arrival_time))

        next_expected_arrival_time = arrival_time + sum(current_diff)/current_length + alpha
        last_arrival_time = arrival_time

    # detection_time = next_expected_arrival_time - enviornment[-1]
    # if detection_time < 0:
    #     detection_time = 0
    # pa = (len(enviornment) - wrong_count) / len(enviornment)
    # cpu_time = psutil.Process(pid).cpu_times().system
    # memory = psutil.Process(pid).memory_info().rss / 1024 / 1024

    return suspicion

def romee2_single(environment, params={'delta_i':1.0, 'alpha':1, 'window_width1':5, 'window_width2':300}):

    delta_i = params['delta_i']
    alpha = params['alpha']
    window_width1 = params['window_width1']
    window_width2 = params['window_width2']
    heartbeat_number = 0

    #mistake_duration = 0
    next_expected_arrival_time = environment[0]
    last_arrival_time = environment[0]
    record = Record(max(params['window_width1'], params['window_width2'])+10)
    #wrong_count = 0
    suspicion = []

    if window_width1 == -1 and window_width2 != -1:
        window_width1, window_width2 = window_width2, window_width1

    if window_width2 == -1:
        '''this means the user is not using two windows, there is only one window'''
        for arrival_time in environment:
            heartbeat_number += 1
            record.append(arrival_time)
            current_length = record.get_length()
            current_sum = record.get_sum(window_width1)

            if arrival_time > next_expected_arrival_time:
                # mistake_duration += arrival_time - next_expected_arrival_time
                # wrong_count += 1
                suspicion.append((heartbeat_number, next_expected_arrival_time,arrival_time, last_arrival_time))

            if window_width1 == -1 or current_length < window_width1:
                '''thie means the user has no requirement on the window width or the current appended records are shorter than the window width, in this case, we use current length to do the calculation.'''
                next_expected_arrival_time = alpha + current_sum / current_length + ((current_length + 1) / 2) * delta_i
                last_arrival_time = arrival_time
            else:
                next_expected_arrival_time = alpha + current_sum / window_width1 + ((window_width1 + 1) / 2) * delta_i
                last_arrival_time = arrival_time

    else:
        '''this is the case that user specifies both the first window width and the second window width'''
        for arrival_time in environment:
            heartbeat_number += 1
            record.append(arrival_time)
            current_length = record.get_length()
            current_sum1 = record.get_sum(window_width1)
            current_sum2 = record.get_sum(window_width2)

            if arrival_time > next_expected_arrival_time:
                # mistake_duration += arrival_time - next_expected_arrival_time
                # wrong_count += 1
                suspicion.append((heartbeat_number, next_expected_arrival_time,arrival_time, last_arrival_time))

            next_expected_arrival_time1 = alpha + current_sum1 / (current_length if (current_length < window_width1) else window_width1) + (((current_length if (current_length < window_width1) else window_width1) + 1) / 2) * delta_i

            next_expected_arrival_time2 = alpha + current_sum2 / (current_length if (current_length < window_width2) else window_width2) + (((current_length if (current_length < window_width2) else window_width2) + 1) / 2) * delta_i

            next_expected_arrival_time = max(next_expected_arrival_time1, next_expected_arrival_time2)
            last_arrival_time = arrival_time

    return suspicion

def suspicion_exchange(suspicion1, trace2, latency_base=1, latency_lower=-0.2, latency_upper=0.2):
    #returns the status list that receiver receivecs, containing the status of Trust or Suspect, and the time receiving the status

    #should not use receiver_local here? should only use the heartbeat list that receiver receives from the monitored node?
    answer_list = {}
    for item in suspicion1:
        heartbeat_number = item[0]
        sus_start = item[1]
        receiver_arrival_time = sus_start + latency_base + random.uniform(latency_lower, latency_upper)
        receiver_heartbeat_time = trace2[heartbeat_number - 1]
        if receiver_heartbeat_time > receiver_arrival_time:
            answer = False
        else:
            answer = True
        sender_arrival_time = receiver_arrival_time + latency_base + random.uniform(latency_lower,latency_upper)

        answer_list[heartbeat_number] = (answer,sender_arrival_time)

    return answer_list

def evaluate(trace1:list, suspicion1:list, response_list:list):
    #print(suspicion1)
    all_FD = len(response_list)
    current_time = 0
    time_list = [current_time]
    result_list = [0]
    suspicion1_index = 0
    last_arrival_time = 0
    error_dict = {}
    detection_time = {}
    if suspicion1 == []:
        return time_list, result_list, error_dict

    print(trace1[-1]+5)
    while current_time < trace1[-1]+5:
        #print(current_time)
        heartbeat_number = suspicion1[suspicion1_index][0]
        next_expected_arrival_time = suspicion1[suspicion1_index][1]
        last_arrival_time = suspicion1[suspicion1_index][-1]
        arrival_time = suspicion1[suspicion1_index][2]

        if current_time < next_expected_arrival_time:
            time_list.append(current_time)
            result_list.append(0)
            #print('writing for increasing current time1', current_time, suspicion1_index)
            current_time = next_expected_arrival_time
            #print(current_time)
            time_list.append(current_time)
            result_list.append(0)
            #print('writing', current_time, suspicion1_index)
            continue

        if current_time >= arrival_time:
            suspicion1_index += 1
            #print('here1',current_time,arrival_time)
            if suspicion1_index >= len(suspicion1):
                print('here2')
                break

        current_time += 0.1
        suspicion_rate1 = 0.5 * (1 - ((next_expected_arrival_time-last_arrival_time)/(current_time-last_arrival_time))**2)
        # if next_expected_arrival_time <= last_arrival_time:
        #     continue

        sus_FD = 0
        current_result = 1
        for l in response_list:
            answer, sender_arrival_time = l[heartbeat_number]
            if sender_arrival_time <= current_time:
                if not answer:
                    sus_FD += 1
                else:
                    current_result = 0
                    break
        if current_result == 0:
            time_list.append(current_time)
            result_list.append(0)
            #print('writing', current_time, suspicion1_index)
        else:
            suspicion_rate2 = 0.5 * ((sus_FD/all_FD)**2)
            time_list.append(current_time)
            if suspicion_rate1 + suspicion_rate2 >= 0.5:
                if heartbeat_number not in detection_time:
                    try:
                        detection_time[heartbeat_number] = current_time - last_arrival_time
                    except:
                        pass

                error_dict[heartbeat_number] = 'yes'
            result_list.append(suspicion_rate1 + suspicion_rate2)
            #print('writing', current_time, suspicion1_index)
    

    return time_list, result_list, error_dict, detection_time

def evaluate2(trace1, suspicion1):
    current_time = 0
    time_list = [current_time]
    result_list = [0]
    suspicion1_index = 0
    last_arrival_time = 0
    error_dict = {}
    detection_time = {}
    if suspicion1 == []:
        return time_list, result_list, error_dict, detection_time

    while current_time < trace1[-1]+5:
        heartbeat_number = suspicion1[suspicion1_index][0]
        next_expected_arrival_time = suspicion1[suspicion1_index][1]
        last_arrival_time = suspicion1[suspicion1_index][-1]
        arrival_time = suspicion1[suspicion1_index][2]
        if current_time < next_expected_arrival_time:
            time_list.append(current_time)
            result_list.append(0)
            current_time = next_expected_arrival_time
            time_list.append(current_time)
            result_list.append(0)
            continue

        if current_time >= arrival_time:
            suspicion1_index += 1
            if suspicion1_index >= len(suspicion1):
                break

        current_time += 0.1
        # if next_expected_arrival_time <= last_arrival_time:
        #     continue

        suspicion_rate1 =(1 - ((next_expected_arrival_time-last_arrival_time)/(current_time-last_arrival_time))**2)
        time_list.append(current_time)
        result_list.append(suspicion_rate1)
        if suspicion_rate1 >= 0.5:
            error_dict[heartbeat_number] = 'yes'
            if heartbeat_number not in detection_time:
                try:
                    detection_time[heartbeat_number] = current_time - last_arrival_time
                except:
                    pass
    
    return time_list, result_list, error_dict, detection_time

if __name__ == '__main__':
    base_trace = naive_trace_generator()
    #print(base_trace)
    suspicion1 = chen_single(base_trace)
    #print(suspicion1)

    trace_list = []
    response_list = []
    for i in range(4):
        trace_list.append(naive_trace_generator())
    #print(trace_list)
    for t in trace_list:
        response_list.append(suspicion_exchange(suspicion1,t,latency_base=2,latency_lower=-1,latency_upper=2))
    #print(response_list)

    time_list, result_list, error_dict, detection_time = evaluate(base_trace, suspicion1, response_list)
    #print(time_list)
    time_list2, result_list2, error_dict2, detection_time2 = evaluate2(base_trace, suspicion1)
    plt.ylim(0,1)
    plt.plot(time_list, result_list, label = 'suspicion_rate to time')
    plt.plot(time_list2,result_list2, color = 'red')
    plt.show()
    print(len(error_dict), len(error_dict2))

    detection_time_sum2 = 0
    for k, v in detection_time2.items():
        detection_time_sum2 += v
    detection_time_avg2 = detection_time_sum2 / len(detection_time2)
    print(detection_time_avg2)
    
    detection_time_sum1 = 0
    for k, v in detection_time.items():
        detection_time_sum1 += v
    detection_time_avg1 = detection_time_sum1 / len(detection_time)
    print(detection_time_avg1)

    
    