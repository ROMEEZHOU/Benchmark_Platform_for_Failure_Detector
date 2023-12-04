#this program supports the one-to-one interaction between 2 failure detectors

import pandas as pd
import numpy as np
import scipy.stats as st
import os
from Extension.record import Record
import random
import matplotlib.pyplot as plt

def get_trace_file_path(receiver:str, sender:str):
    tarce_file_path = os.path.join('data2','Node'+receiver,'trace'+sender+'.csv')
    return tarce_file_path

def chen_single(suspector:str, suspected:str, params={'delta_i':100000000.0, 'n':20, 'alpha':1000000}):
    #returns the suspicion list that suspector generates
    #also returns the quality of local fd?
    delta_i = params['delta_i']
    n = params['n']
    alpha = params['alpha']
    heartbeat_number = 0

    trace_file_path = get_trace_file_path(suspector, suspected)
    df = pd.read_csv(trace_file_path)
    environment = np.array(df.timestamp_receive)[:200]

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


def bertier_single(suspector:str, suspected:str, params={'delta_i':100000000.0, 'n':100, 'delay':0, 'var':0, 'gamma':0.01, 'beta':1, 'phi':5.5}):

    delta_i = params['delta_i']
    n = params['n']
    delay = params['delay']
    var = params['var']
    gamma = params['gamma']
    beta = params['beta']
    phi = params['phi']
    heartbeat_number = 0

    trace_file_path = get_trace_file_path(suspector, suspected)
    df = pd.read_csv(trace_file_path)
    environment = np.array(df.timestamp_receive)[:200]
    suspicion = []

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

def accrual_single(suspector:str, suspected:str, params={'delta_i':100000000.0, 'n':1000, 'phi':10}):
    delta_i = params['delta_i']
    n = params['n']
    phi = params['phi']
    heartbeat_number = 0

    trace_file_path = get_trace_file_path(suspector, suspected)
    df = pd.read_csv(trace_file_path)
    environment = np.array(df.timestamp_receive)[:200]

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


def romee_shitty_single(suspector:str, suspected:str, params ={'delta_i':100000000.0, 'n':200, 'alpha':7000000}):
    delta_i = params['delta_i']
    n = params['n']
    alpha = params['alpha']
    heartbeat_number = 0
    
    trace_file_path = get_trace_file_path(suspector, suspected)
    df = pd.read_csv(trace_file_path)
    environment = np.array(df.timestamp_receive)[:200]

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

def romee2_single(suspector:str, suspected:str, params={'delta_i':100000000.0, 'alpha':5000000, 'window_width1':5, 'window_width2':300}):

    delta_i = params['delta_i']
    alpha = params['alpha']
    window_width1 = params['window_width1']
    window_width2 = params['window_width2']
    heartbeat_number = 0

    trace_file_path = get_trace_file_path(suspector, suspected)
    df = pd.read_csv(trace_file_path)
    environment = np.array(df.timestamp_receive)[:200]

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

def get_suspicion_single(suspector:str, suspected:str, algorithm:str):
    #returns the suspicion list that suspector generates based on the assigned algorithm
    alg_to_func = {'chen':chen_single, 'bertier':bertier_single, 'accrual':accrual_single, 'romee_shitty':romee_shitty_single, 'romee2':romee2_single}
    algorithm = algorithm.lower()
    if algorithm in alg_to_func:
        return alg_to_func[algorithm](suspector, suspected)

def suspicion_exchange(sender:str, receiver:str, suspected:str, sender_local:list, latency = 0):
    #returns the status list that receiver receivecs, containing the status of Trust or Suspect, and the time receiving the status

    #should not use receiver_local here? should only use the heartbeat list that receiver receives from the monitored node?

    file_path1 = get_trace_file_path(receiver, sender)
    df1 = pd.read_csv(file_path1)
    sender_send = np.array(df1.timestamp_send)
    receiver_receive = np.array(df1.timestamp_receive)
    file_path2 = get_trace_file_path(sender, receiver)
    df2 = pd.read_csv(file_path2)
    receiver_send = np.array(df2.timestamp_send)
    sender_receive = np.array(df2.timestamp_receive)

    receiver_suspected_file = get_trace_file_path(receiver, suspected)
    df = pd.read_csv(receiver_suspected_file)
    receiver_environment = np.array(df.timestamp_receive)[:200]

    #for each suspicion in the sender's local list, send a request to the receiver for its answer

    sender_send_index = 0
    receiver_send_index = 0
    answer_list = {} #stores receiver's answer and the time sender receives the answer (in sender's local time)
    for item in sender_local:
        heartbeat_number = item[0]
        sus_start = item[1]
        receiver_receive_min = 0
        receiver_receive_max = 0
        sender_receive_min = 0
        sender_receive_max = 0

        # while sus_start <= sender_send[sender_send_index]:
        #     receiver_receive_min = receiver_receive[sender_send_index]
        #     sender_send_index += 1
        # receiver_receive_max = receiver_receive[sender_send_index]

        sender_send_base = sender_send[sender_send_index]
        receiver_receive_base = receiver_receive[sender_send_index]
        receiver_receive_final = sus_start - sender_send_base + receiver_receive_base


        if receiver_environment[heartbeat_number -1] > receiver_receive_final:
            answer = False 
            #when sender's request comes, receiver hasn't got the corresponding heartbeat
        else:
            answer = True

        #print(answer, sus_start, receiver_receive_final, sender_send_index)

        receiver_send_base = receiver_send[receiver_send_index]
        sender_receive_base = sender_receive[receiver_send_index]
        sender_receive_final = receiver_receive_final - receiver_send_base + sender_receive_base
        sender_receive_final = max(sender_receive_final, sus_start)
        
        # while receiver_receive_final <= receiver_send[receiver_send_index]:
        #     #print("receiver send index",receiver_send_index)
        #     sender_receive_min = sender_receive[receiver_send_index]
        #     receiver_send_index += 1
        # sender_receive_final = min(sender_receive_min, sus_start)

        answer_list[heartbeat_number] = (answer,sender_receive_final)
    print(answer_list)
        #now determine the answer receiver responds

    return answer_list
#seems like there is nothing wrong with this function

def evaluate1(suspector, suspected, suspicion1, response_list):
    original_trace_file = get_trace_file_path(suspector, suspected)
    df = pd.read_csv(original_trace_file)
    environment = np.array(df.timestamp_receive)[:200]
    time_begin = environment[0]
    time_end = environment[-1] + 10000000
    current_time = 0
    time_list = [current_time]
    result_list = [0]
    suspicion1_index = 0
    all_FD = len(response_list)
    error_dict = {}
    detection_time = {}

    while current_time < time_end - time_begin:
        heartbeat_number = suspicion1[suspicion1_index][0]
        expected_arrival_time = suspicion1[suspicion1_index][1]
        arrival_time = suspicion1[suspicion1_index][2]
        last_arrival_time = suspicion1[suspicion1_index][-1]

        if time_begin+current_time < expected_arrival_time:
            time_list.append(current_time)
            result_list.append(0)
            current_time = expected_arrival_time - time_begin
            time_list.append(current_time)
            result_list.append(0)
            print(heartbeat_number, expected_arrival_time, last_arrival_time, arrival_time)
            continue

        if time_begin+current_time > arrival_time:
            suspicion1_index += 1
            if suspicion1_index >= len(suspicion1):
                break

        current_time += 1000000
        print(time_begin, current_time, last_arrival_time)
        print((expected_arrival_time-last_arrival_time))
        suspicion_rate1 = 0.5 * (1 - ((expected_arrival_time-last_arrival_time)/(time_begin+current_time-last_arrival_time))**2)


        sus_FD = 0
        current_result = 1
        for l in response_list:
            answer, sender_arrival_time = l[heartbeat_number]
            if sender_arrival_time <= time_begin+current_time:
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
            print('suspicion rate',suspicion_rate1, suspicion_rate2)
            if suspicion_rate1 + suspicion_rate2 >= 0.5:
                if heartbeat_number not in detection_time:
                    try:
                        detection_time[heartbeat_number] = time_begin + current_time - last_arrival_time
                    except:
                        pass

                error_dict[heartbeat_number] = 'yes'
            result_list.append(suspicion_rate1 + suspicion_rate2)
            #print('writing', current_time, suspicion1_index)

    return time_list, result_list, error_dict, detection_time

def evaluate2(suspector, suspected, suspicion1):
    original_trace_file = get_trace_file_path(suspector, suspected)
    df = pd.read_csv(original_trace_file)
    trace1 = np.array(df.timestamp_receive)[:200]
    current_time = 0
    time_list = [current_time]
    result_list = [0]
    time_begin = trace1[0]
    time_end = trace1[-1] + 10000000
    suspicion1_index = 0
    last_arrival_time = 0
    error_dict = {}
    detection_time = {}
    #print('time begin', time_begin, 'time end', time_end)
    if suspicion1 == []:
        return time_list, result_list, error_dict, detection_time
    
    while time_begin+current_time < time_end:
        heartbeat_number = suspicion1[suspicion1_index][0]
        expected_arrival_time = suspicion1[suspicion1_index][1]
        last_arrival_time = suspicion1[suspicion1_index][-1]
        arrival_time = suspicion1[suspicion1_index][2]
        print(heartbeat_number, expected_arrival_time, last_arrival_time, arrival_time)

        if time_begin+current_time < expected_arrival_time:
            time_list.append(current_time)
            result_list.append(0)
            current_time = expected_arrival_time - time_begin
            time_list.append(current_time)
            result_list.append(0)
            continue

        if time_begin + current_time >= arrival_time:
            suspicion1_index += 1
            if suspicion1_index >= len(suspicion1):
                print('here2')
                break

        current_time += 1000000

        suspicion_rate1 =(1 - ((expected_arrival_time-last_arrival_time)/(time_begin+current_time-last_arrival_time))**2)
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
    suspected = '0'
    sender = '1'
    co_suspector1 = '3'
    co_suspector2 = '5'
    co_suspector3 = '6'
    co_suspector4 = '7'

    suspicion1 = accrual_single(sender, suspected)
    print(suspicion1)
    #seems correct so far
    response_list = []
    for c_s in ['3','5','6','7']:
        response_list.append(suspicion_exchange(sender, c_s, suspected, suspicion1))
    time_list1, result_list1, error_dict1, detection_time1 = evaluate1(sender, suspected, suspicion1, response_list)
    print(result_list1)
    time_list2, result_list2, error_dict2, detection_time2 = evaluate2(sender, suspected, suspicion1)

    plt.ylim(0,0.1)
    plt.plot(time_list1, result_list1, label = 'suspicion_rate to time')
    plt.plot(time_list2,result_list2, color = 'red')
    plt.show()
    print(len(error_dict1), len(error_dict2))

    detection_time_sum2 = 0
    for k, v in detection_time2.items():
        detection_time_sum2 += v
    detection_time_avg2 = detection_time_sum2 / len(detection_time2)
    print(detection_time_avg2)
    
    detection_time_sum1 = 0
    for k, v in detection_time1.items():
        detection_time_sum1 += v
    detection_time_avg1 = detection_time_sum1 / len(detection_time1)
    print(detection_time_avg1)


