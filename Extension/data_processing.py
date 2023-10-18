'''
This program is for processing the traces to form large-scale distributed system traces
'''

'''
Basic logics:
when we add a node 11, the traces other node received from node 11 is generated from the traces they received from node 1, with a reshape window number of 3 and some noices
for node 1, traces it received from node 11 is generated from the traces it received from node 2, with a reshape window number of 4 and some noices

For the traces that node 11 received, they are generated from the traces that node 1 received from other nodes, with some noices
For the traces that node 11 received from node 2, they are generated from traces node 2 received from node 1, with some noices
'''

'''
A basic function for doing permutation on 1 to n
'''

import pandas as pd
import os

def permute(n):
    base_list = []
    for i in range(n):
        base_list.append(i+1)
    return permute_list(base_list)

def permute_list(list):
    result = []
    if len(list) == 0:
        return [[]]
    if len(list) == 1:
        return [list]
    for i in range(len(list)):
        current = [list[i]]
        remain = list[:i] + list[i+1:]
        for sub_list in permute_list(remain):
            result.append(sub_list + current)
    return result

'''
Function for adding one node traces
'''
'''
Basic logics:
when we add a node 10, the traces other node received from node 10 is generated from the traces they received from node 0, with a reshape window number of 3 and some noices
for node 0, traces it received from node 10 is generated from the traces it received from node 1, with a reshape window number of 4 and some noices

For the traces that node 10 received, they are generated from the traces that node 0 received from other nodes, with some noices
For the traces that node 10 received from node 1, they are generated from traces node 1 received from node 0, with some noices
'''

def add_node(data_file, permute_number, permute_list, permute_index, backup_permute_list, backup_permute_index, node_number):
    original_list = [0,1,3,5,6,7,8,9]
    copied_node_number = original_list[(node_number - 2) % 8]
    copied_node_path = os.path.join(data_file, 'Node'+str(copied_node_number))
    '''the data misses node 2 and node 4'''

    '''first step, for other node, generate trace10.csv from trace0.csv'''
    for item in os.listdir(data_file):
        if item[0:4] == 'Node' and item != 'Node' + str(copied_node_number):
            csv_path = os.path.join(data_file, item, 'trace' + str(copied_node_number) + '.csv')

            df = pd.read_csv(csv_path, usecols=['site', 'timestamp_send', 'timestamp_receive'])
            first_timestamp_send, first_timestamp_receive = df.loc[0, ['timestamp_send', 'timestamp_receive']]
            df_diff = df.diff()[1:]
            print('df at first:', df_diff.shape)
            
            new_trace = pd.DataFrame(data = {'site': [node_number], 'timestamp_send': [first_timestamp_send], 'timestamp_receive': [first_timestamp_receive]})
            total_length, _ = df_diff.shape
            sub_length = total_length // permute_number
            print(sub_length)
            for per_index in permute_list[permute_index]:
                if per_index != permute_number:
                    print(((per_index-1) * sub_length), (per_index*sub_length) + 1)
                    new_trace = pd.concat([new_trace, df_diff.iloc[((per_index-1) * sub_length):(per_index*sub_length),:]])
                    print(new_trace.shape)
                else:
                    new_trace = pd.concat([new_trace, df_diff.iloc[((permute_number-1)*sub_length):,:]])
                    print((total_length - (permute_number-1)*sub_length),'end')
                    print(new_trace.shape)

            '''write the new trace'''
            new_trace = new_trace.cumsum()
            print(new_trace.shape)
            new_trace.to_csv(path_or_buf=os.path.join(data_file, item, 'trace' + str(node_number) + '.csv'), index=False)


        elif item[0:4] == 'Node':
            '''if the node is the copied node itself, copy the traces it received from Node copied_node_number+1 as the traces it received from Node node_number'''

            backup_copied_node_number = original_list[(node_number - 1) % 8]
            csv_path = os.path.join(data_file, item, 'trace' + str(backup_copied_node_number) + '.csv')

            df = pd.read_csv(csv_path, usecols=['site', 'timestamp_send', 'timestamp_receive'])
            first_timestamp_send, first_timestamp_receive = df.loc[0, ['timestamp_send', 'timestamp_receive']]
            df_diff = df.diff()[1:]

            new_trace = pd.DataFrame(data = {'site': [1], 'timestamp_send': [first_timestamp_send], 'timestamp_receive': [first_timestamp_receive]})
            total_length, _ = df_diff.shape
            sub_length = total_length // (permute_number+1)
            for per_index in backup_permute_list[backup_permute_index]:
                if per_index != permute_number+1:
                    new_trace = pd.concat([new_trace, df_diff.iloc[((per_index-1) * sub_length):(per_index*sub_length),:]])
                else:
                    new_trace = pd.concat([new_trace, df_diff.iloc[((permute_number-1)*sub_length):,:]])
            new_trace = new_trace.cumsum()
            new_trace.to_csv(path_or_buf=os.path.join(data_file, item, 'trace' + str(node_number) + '.csv'), index=False)


    '''second step, for node10, generate tracex.csv from tracex.csv in Node 0'''
    add_node_path = os.path.join(data_file, 'Node'+str(node_number))
    os.makedirs(add_node_path, exist_ok=True)
    for item in os.listdir(copied_node_path):
        if item[0:5] != 'trace':
            continue
        if item == 'trace'+str(node_number)+'.csv':
            continue
        csv_path = os.path.join(copied_node_path, item)
        print(csv_path)
        df = pd.read_csv(csv_path, usecols=['site', 'timestamp_send', 'timestamp_receive'])
        first_timestamp_send, first_timestamp_receive = df.loc[0, ['timestamp_send', 'timestamp_receive']]
        df_diff = df.diff()[1:]

        site = str(item).split('.')[0][5]
        site = int(site)
        new_trace = pd.DataFrame(data = {'site': [site], 'timestamp_send': [first_timestamp_send], 'timestamp_receive': [first_timestamp_receive]})
        total_length, _ = df_diff.shape
        sub_length = total_length // permute_number
        for per_index in permute_list[permute_index]:
            if per_index != permute_number:
                new_trace = pd.concat([new_trace, df_diff.iloc[((per_index-1) * sub_length):(per_index*sub_length),:]])
            else:
                new_trace = pd.concat([new_trace, df_diff.iloc[((permute_number-1)*sub_length):,:]])
        new_trace = new_trace.cumsum()
        new_trace.to_csv(path_or_buf=os.path.join(add_node_path, item), index=False)

    '''now generate the trace node10 received from the node it copies'''
    backup_copied_node_number = original_list[(node_number + 1) % 8]
    csv_path = os.path.join(data_file, 'Node'+str(backup_copied_node_number), 'trace'+str(copied_node_number)+'.csv')

    df = pd.read_csv(csv_path, usecols=['site', 'timestamp_send', 'timestamp_receive'])
    first_timestamp_send, first_timestamp_receive = df.loc[0, ['timestamp_send', 'timestamp_receive']]
    df_diff = df.diff()[1:]

    new_trace = pd.DataFrame(data = {'site': [copied_node_number], 'timestamp_send': [first_timestamp_send], 'timestamp_receive': [first_timestamp_receive]})
    total_length, _ = df_diff.shape
    sub_length = total_length // (permute_number+1)
    for per_index in backup_permute_list[backup_permute_index]:
        if per_index != permute_number+1:
            new_trace = pd.concat([new_trace, df_diff.iloc[((per_index-1) * sub_length):(per_index*sub_length),:]])
        else:
            new_trace = pd.concat([new_trace, df_diff.iloc[( (permute_number-1)*sub_length):,:]])
    new_trace = new_trace.cumsum()
    new_trace.to_csv(path_or_buf=os.path.join(add_node_path, 'trace'+str(copied_node_number)+'.csv'), index=False)

    return

'''
main function for controlling data processing, the number n is the number of nodes we want to add to existing traces
'''
def add_data(n, data_file='../data2'):
    added_node = 0
    permute_number = 3
    permute_list = permute(permute_number)
    backup_permute_list = permute(permute_number + 1)
    permute_upper = added_node + len(permute_list)
    permute_index = 0
    backup_permute_index = 0
    node_number = 10

    while added_node < n:
        if added_node >= permute_upper:
            permute_number += 2
            permute_list = permute(permute_number)
            backup_permute_list = permute(permute_number+1)
            permute_index = 0
            backup_permute_index = 0
            permute_upper += len(permute_list)

        add_node(data_file = data_file, permute_number = permute_number, permute_list = permute_list, permute_index = permute_index, backup_permute_list = backup_permute_list, backup_permute_index = backup_permute_index, node_number = node_number)
        last_round = added_node // 8
        added_node += 1
        node_number += 1
        current_round = added_node // 8
        if current_round > last_round:
            permute_index += 1
            backup_permute_index += 1


'''The function to re-arrange the original Node traces into pairs, e.g. under data2/Node0, we have trace1.csv for the heartbeats received from Node1'''
def first_operation():
    directories = []
    for item in os.listdir("../data"):
        item_path = os.path.join("../data", item)
        if os.path.isdir(item_path):
            directories.append(item)
    for i in directories:
        for j in directories:
            if i != j:
                #create multi-processing based on different nodes
                node_path = os.path.join("../data", i)
                csv_path = os.path.join(node_path, 'trace.csv')
                df = pd.read_csv(csv_path)
                #print(j)
                df = df[df.site == int(j[4:])]
                df.to_csv(path_or_buf=os.path.join('../data2', i, 'trace' + j[4:] + '.csv'), index=False)

if __name__ == '__main__':
    add_data(13,'..\data2')