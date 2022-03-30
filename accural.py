import pandas as pd
import numpy as np
from record import Record
import matplotlib.pyplot as plt
import scipy.stats as st

def accural_estimate_for_single_value(enviornment, delta_i, n, phi):
    mistake_duration = 0
    expected_arrival_time = enviornment[0]
    record = Record(n)

    for arrival_time in enviornment:
        record.append(arrival_time)
        difference = record.get_difference()

        if arrival_time > expected_arrival_time:
            mistake_duration += arrival_time - expected_arrival_time

        # Then start to calculate the expected arrival time
        if len(difference) == 0:
            # Means there are only one arrival time in the record
            expected_arrival_time = arrival_time + delta_i
        elif len(difference) == 1:
            # Means there are only two arrival time in the record
            expected_arrival_time = arrival_time + difference[0]
        else:
            mean = np.mean(difference)
            scale = np.std(difference)
            expected_interval = st.norm.ppf(1 - np.power(0.1, phi), loc=mean, scale=scale)
            expected_arrival_time = arrival_time + expected_interval
    return mistake_duration

def accural_estimate_for_phi_array(enviornment, delta_i, n, phi_array):
    length = len(phi_array)
    mistake_duration = np.zeros(length, dtype=float)
    expected_arrival_time = np.array([enviornment[0] for i in range(length)], dtype=float)
    record = Record(n)

    for arrival_time in enviornment:
        record.append(arrival_time)
        difference = record.get_difference()

        duration = -expected_arrival_time + arrival_time
        duration = np.maximum(duration, 0)
        mistake_duration += duration

        # Then start to calculate the expected arrival time
        if len(difference) == 0:
            # Means there are only one arrival time in the record
            expected_arrival_time = arrival_time + delta_i
        elif len(difference) == 1:
            # Means there are only two arrival time in the record
            expected_arrival_time = arrival_time + difference[0]
        else:
            mean = np.mean(difference)
            scale = np.std(difference)
            expected_interval = st.norm.ppf(1 - np.power(0.1, phi_array), loc=mean, scale=scale)
            expected_arrival_time = arrival_time + expected_interval
    return mistake_duration

def accural_estimate_for_n_array(enviornment, delta_i, n_array, phi):
    length = len(n_array)
    mistake_duration = np.zeros(length, dtype=float)
    expected_arrival_time = np.array([enviornment[0] for i in range(length)], dtype=float)
    record_list = [Record(i) for i in n_array]
    interval_list = np.zeros(length, dtype=float)

    for arrival_time in enviornment:
        for inx, record in enumerate(record_list):
            record.append(arrival_time)
            difference = record.get_difference()

            # Then start to calculate the expected arrival time
            if len(difference) == 0:
                # Means there are only one arrival time in the record
                expected_interval = delta_i
            elif len(difference) == 1:
                # Means there are only two arrival time in the record
                expected_interval = difference[0]
            else:
                mean = np.mean(difference)
                scale = np.std(difference)
                expected_interval = st.norm.ppf(1 - np.power(0.1, phi), loc=mean, scale=scale)

            interval_list[inx] = expected_interval

        duration = -expected_arrival_time + arrival_time
        duration = np.maximum(duration, 0)
        mistake_duration += duration

        expected_arrival_time = arrival_time + interval_list
    return mistake_duration

def accural_estimate(enviornment, delta_i, n, phi):
    if type(n) != np.ndarray and type(n) != int:
        raise TypeError('The data type of n can only be numpy array or int')
    if type(phi) != np.ndarray and type(phi) != int:
        raise TypeError('The data type of alpha can only be numpy array or int')

    if type(n) == np.ndarray and type(phi) == np.ndarray:
        raise TypeError('The data type of n and alpha cannot be both array')
    elif type(phi) == np.ndarray:
        mistake_duration = accural_estimate_for_phi_array(enviornment, delta_i, n, phi)
    elif type(n) == np.ndarray:
        mistake_duration = accural_estimate_for_n_array(enviornment, delta_i, n, phi)
    else:
        mistake_duration = accural_estimate_for_single_value(enviornment, delta_i, n, phi)
    return mistake_duration

if __name__ == '__main__':
    df = pd.read_csv(r'.\data\Node0\trace.csv')
    df = df[df.site == 8]
    arrival_time_array = np.array(df.timestamp_receive)

    delta_i = 100000000.0
    # n = 1000
    n = np.array([2, 10, 100, 1000, 10000])
    phi = 1
    # phi = np.array([i for i in range(1000)])
    # phi = phi / 100

    mistake_duration = accural_estimate(arrival_time_array, delta_i, n, phi) / 1000000000

    # print(mistake_duration)

    plt.plot(n, mistake_duration)
    plt.show()



