#!/usr/bin/env python
# coding: utf-8


from model.sfcn import SFCN
import numpy as np
import pandas as pd
from volumedatagenerator import VolumeDataGeneratorRegression
import matplotlib.pyplot as plt

import time
import sys

def train_and_evaluate(idx, only_evaluate=False):
    name = 'sfcn_pyramid_nopool'
    index=int(idx)

    batch_size = 8
    gpu_list = [4, 5, 6, 7]
    cpu_workers = 8
    epochs_num = 64

    idps_labels = pd.read_csv('csv/idps_desc.csv')['id'].to_list()
    idps_labels = [str(l) for l in idps_labels]

    train_df = pd.read_csv('csv/split_train.csv', index_col='id').dropna()
    valid_df = pd.read_csv('csv/split_valid.csv', index_col='id').dropna()
    test_df = pd.read_csv('csv/split_test.csv', index_col='id').dropna()

    input_dim = [182, 218, 182]
    num_output = len(idps_labels)

    train_gen = VolumeDataGeneratorRegression(
        sample_df=train_df, 
        batch_size=batch_size, 
        #num_reg_classes=num_output, 
        dim=input_dim,
        output_preprocessing='quantile', 
        idps_labels=idps_labels)

    scaler_instance = train_gen.get_scaler_instance()

    valid_gen = VolumeDataGeneratorRegression(
        sample_df=valid_df, 
        batch_size=batch_size, 
        #num_reg_classes=num_output, 
        dim=input_dim,
        output_scaler=scaler_instance,
        shuffle=False, 
        idps_labels=idps_labels)

    model = SFCN(
        input_dim=[182, 218, 182, 1], 
        output_dim=num_output,
        conv_num_filters=[32, 64, 64, 128, 256, 256], 
        conv_kernel_sizes=[3, 3, 3, 3, 1, 1], 
        conv_strides=[3, 3, 3, 3, 1, 1],
        conv_padding=['same', 'same', 'same', 'same', 'same', 'valid'],
        pooling_size=[2, 2, 2, 2, 2],
        pooling_type=['no_pool', 'no_pool', 'no_pool', 'no_pool', 'no_pool'],
        batch_norm=True,
        dropout=False,
        softmax=False,
        use_float16=True,  
        reduce_lr_on_plateau=0.5,
        gpu_list = gpu_list,
        name=name+'_'+str(index))

    if not only_evaluate:
        start = time.time()
        model.compile(learning_rate=1e-3)
        model.train_generator(train_gen, valid_gen, batch_size=batch_size, epochs=epochs_num, workers=cpu_workers)
        stop = time.time()

        time_elapsed = stop - start
        print('time elapsed (hours): {}'.format(time_elapsed/(3600)))
        
    # validation set
    model.load_weights('weights/checkpoint_' + name + '_' + str(index))
    # model.evaluate_generator(valid_gen, batch_size, filename=name + '_val', workers=cpu_workers)

    # test set
    test_gen = VolumeDataGeneratorRegression(
        sample_df=test_df, 
        batch_size=batch_size, 
        #num_reg_classes=num_output, 
        dim=input_dim,
        output_scaler=scaler_instance,
        shuffle=False, 
        idps_labels=idps_labels
    )

    model.evaluate_generator(test_gen, batch_size, filename=name + '_test', workers=cpu_workers)

if __name__=='__main__':
    # for i in range(int(sys.argv[1])): 
    #     main(i)
    train_and_evaluate(sys.argv[1], only_evaluate=False)
    #train_and_evaluate(1, only_evaluate=True)
