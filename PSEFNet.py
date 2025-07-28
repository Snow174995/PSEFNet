import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import multiprocessing
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fast.trainer import train, evaluate, to_string, predict, count_parameters
from fast.preprocess import h_step_ahead_split, h_horizons_ahead_split, \
    split_data_to_train_test, split_data_to_train_val_test, MinMaxScale, StandardScale, Scale, MaxScale
from model import PSEFNet




def main():

    data_type = torch.float32

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # -1: cpu; 0: cuda:0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    mts_dir = r'D:/workspace/pyFAST-new/pyFAST/'
    excel_file1 = mts_dir + r'Mooc_dataset_prepared.csv'


    target_df = pd.read_excel(excel_file1)
    target_df = target_df.values[2, 1:]
    course_id = target_df.values[0, 1:]
    sentiment_label = target_df.values[5, 1:]
    grade_score = target_df.values[3, 1:]
    course_rating = target_df.values[4, 1:]

    m, n = len(target_df), len(target_df[0])

    target_array = np.array(target_df, dtype=np.float64)
    course_id = np.array(course_id, dtype=np.float64)
    sentiment_label = np.array(sentiment_label, dtype=np.float64)
    grade_score = np.array(grade_score, dtype=np.float64)
    course_rating = np.array(course_rating, dtype=np.float64)


    batch_size = 64
    kernel_size = 9
    num_nodes = len(target_array[1])
    max_input_window_size = 25
    in_channels = 10
    out_channels = 1
    K = 5
    input_window_size = 12 # a.k.a., look-back window
    multi_layer = 10
    num_layers = 1
    hidden_size = 32
    num_block = 2
    embed_dim = 10
    dropout = 0.1
    leaky_rate = 0.2
    output_window_size = 12  # a.k.a., multiple horizons
    cheb_k = 3
    start_window_size, end_window_size = input_window_size, input_window_size   # 1, max_window_size
    stride = 1
    total_size = (len(target_array) - input_window_size - output_window_size) // stride + 1
    test_size = round(total_size * 0.2)
    seq_len = 128


    # adj_array = torch.tensor(adj_array, dtype=data_type).to(device)

    target_array = torch.tensor(target_array, dtype=data_type).to(device)
    scaler = StandardScale().fit(target_array)
    normalized_array = scaler.transform(target_array)
    # normalized_array = torch.tensor(normalized_array, dtype=data_type).to(device)
    normalized_array = torch.tensor(normalized_array, dtype=data_type).to(device)


    for _lag in range(start_window_size, end_window_size + 1):
        # samples, targets = h_step_ahead_split(normalized_array, input_window_size, horizon, stride=1)
        samples, targets = h_horizons_ahead_split(normalized_array, input_window_size, output_window_size, stride)
        # x_train, y_train, x_val, y_val, x_test, y_test = \
        # split_data_to_train_val_test(samples, targets, val_size, test_size)
        x_train, y_train, x_test, y_test = split_data_to_train_test(samples, targets, test_size // stride)
        # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        model = PSEFNet(x_train.shape[1], x_train.shape[2], y_train.shape[1],
                    dropout=0.1,course_id, sentiment_label, grade_score, course_rating) # default lr=0.00001, optimal lr=0.0002

        print('total/test,{}/{},#parameters,{}'.format(total_size, test_size, count_parameters(model)[1]))

        model_name = type(model).__name__
        model = model.to(device, dtype=data_type)
        if torch.cuda.device_count() > 1 and device.type == 'cuda':
            model = nn.DataParallel(model)
        else:

            torch.set_num_threads(multiprocessing.cpu_count() - 2)

        train(model, [x_train, y_train], validation_data=[x_test, y_test],
              epochs=20, batch_size=batch_size, stride=stride,
              lr=0.001, lr_decay=0.05, lr_decay_step_size=50, weight_decay=0.,
              shuffle=True, verbose=True,
              normalization_scaler=scaler, display_interval=50, checkpoint=None, L=[])


        y_hat = predict(model, [x_test], batch_size)
        predict_values = scaler.inverse_transform(y_hat)
        real_values = scaler.inverse_transform(y_test)
        results = evaluate(real_values, predict_values)
        print(model_name + '\t' + to_string(_lag, *results))
        # P_10 = predict_values[:, :, 22]
        # T_10 = real_values[:, :, 22]
        # P = P_10[2880:4896, 6]
        # T = T_10[2880:4896, 6]
        # df1 = pd.DataFrame({'T': T})
        # df2 = pd.DataFrame({'P': P})
        # df1.to_csv('D:/workspace/pyFAST-new/pyFAST/traffic_data/data/PEMSD8/T_column.csv', index=False)
        # df2.to_csv('D:/workspace/pyFAST-new/pyFAST/traffic_data/data/PEMSD8/P_column.csv', index=False)
        #
        # # time_steps = pd.date_range("2016-8-02 09:00", periods=2016, freq="5min")
        # # 绘制真实值和预测值的曲线图
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(len(T)), T, label='True Values', color='blue', linestyle='-', linewidth=1)
        # plt.plot(range(len(T)), P, label='Predicted Values', color='red', linestyle='-', linewidth=1)
        #
        # # 添加标题和标签
        # # plt.title(f'True vs Predicted Values for Node {10}')
        # plt.xlabel('Time Step (5-minute interval)', fontsize=14)
        # plt.ylabel('Traffic Flow (vehicles / 5 min)', fontsize=14)
        # plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False)
        #
        #
        # # 显示图形
        # plt.grid(True)
        # plt.show()



        # print(P)
        # print(T)

    print('hello')



if __name__ == '__main__':
    main()
