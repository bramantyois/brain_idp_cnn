#!/usr/bin/env python
# coding: utf-8


from res_sfcn import ResSFCN
import numpy as np
import pandas as pd
from volumedatagenerator import VolumeDataGeneratorRegression
import matplotlib.pyplot as plt

import sys

def main(idx):
    name = 'sfcn_res_avg'
    index=int(idx)

    batch_size = 8
    gpu_num = 8
    cpu_workers = 8
    epochs_num = 64

    train_df = pd.read_csv('csv/split_train.csv', index_col='eid').dropna()
    valid_df = pd.read_csv('csv/split_valid.csv', index_col='eid').dropna()
    test_df = pd.read_csv('csv/split_test.csv', index_col='eid').dropna()

    input_dim = [182, 218, 182]
    num_output = len(train_df.columns)-1

    train_gen = VolumeDataGeneratorRegression(
        sample_df=train_df, 
        batch_size=batch_size, 
        #num_reg_classes=num_output, 
        dim=input_dim,
        output_preprocessing='quantile')

    scaler_instance = train_gen.get_scaler_instance()

    valid_gen = VolumeDataGeneratorRegression(
        sample_df=valid_df, 
        batch_size=batch_size, 
        #num_reg_classes=num_output, 
        dim=input_dim,
        output_scaler=scaler_instance,
        shuffle=False)

    test_gen = VolumeDataGeneratorRegression(
        sample_df=test_df, 
        batch_size=batch_size, 
        #num_reg_classes=num_output, 
        dim=input_dim,
        output_scaler=scaler_instance,
        shuffle=False
    )

    model = ResSFCN(
            input_dim=[182, 218, 182, 1], 
            output_dim=num_output,
            conv_num_filters=[32, 64, 64, 128, 256, 256], 
            conv_kernel_sizes=[3, 3, 3, 3, 3, 1], 
            conv_strides=[1, 1, 1, 1, 1, 1],
            conv_padding=['same', 'same', 'same', 'same', 'same', 'valid'],
            pooling_size=[2, 2, 2, 2, 2],
            pooling_type=['avg_pool', 'avg_pool', 'avg_pool', 'avg_pool', 'avg_pool'],
            batch_norm=True,
            dropout=False,
            softmax=False,
            gpu_num=gpu_num,
            use_float16=True,
            name=name+'_'+str(index)
            )
            
    model.compile(learning_rate=3e-4)

    model.train_generator(train_gen, valid_gen, batch_size=batch_size, epochs=epochs_num, workers=cpu_workers)

    model.load_weights('weights/checkpoint_' + name + '_' + str(index))
    model.evaluate_generator(valid_gen, batch_size, filename=name + '_val', workers=cpu_workers)
    model.evaluate_generator(test_gen, batch_size, filename=name + '_test', workers=cpu_workers)

if __name__=='__main__':
    for i in range(int(sys.argv[1])): 
        main(i)