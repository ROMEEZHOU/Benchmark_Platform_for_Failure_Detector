import matplotlib.pyplot as plt
import pandas as pd

def trace_visualization(trace_file):
    df1 = pd.read_csv(trace_file, usecols=['timestamp_send'])
    df2 = pd.read_csv(trace_file, usecols=['timestamp_receive'])
    df1_diff = df1.diff()
    df2_diff = df2.diff()

    df1_diff.plot()
    df2_diff.plot()

if __name__ == "__main__":
    trace_visualization('../data2/Node3/trace5.csv')
    plt.show()