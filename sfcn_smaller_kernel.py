#!/usr/bin/env python
# coding: utf-8


from model.sfcn import SFCN
from keras import backend as K

import numpy as np
import pandas as pd
from volumedatagenerator import VolumeDataGeneratorRegression
import matplotlib.pyplot as plt

import time
import sys

def sfcn_pyramid_small_kern(idx, only_evaluate=False):
    name = 'sfcn_pyramid_small_kern'
    index=int(idx)

    batch_size = 8
    gpu_list =  [4,5,6,7]
    cpu_workers = 8
    epochs_num = 64
    input_preprocess = 'standardize'

    idps_labels = pd.read_csv('csv/idps_desc.csv')['id'].to_list()
    idps_labels = [str(l) for l in idps_labels]

    train_df = pd.read_csv('csv/split_train.csv', index_col='id').dropna()
    train_stats = pd.read_csv('csv/train_stats.csv', index_col='id')

    valid_df = pd.read_csv('csv/split_valid.csv', index_col='id').dropna()

    input_dim = [160, 192, 160]
    num_output = len(idps_labels)

    model = SFCN(
        input_dim=[160, 192, 160, 1], 
        output_dim=num_output,
        conv_num_filters=[32, 64, 64, 128, 256, 256], 
        conv_kernel_sizes=[2, 2, 2, 2, 2, 1], 
        conv_strides=[1, 1, 1, 1, 1, 1],
        conv_padding=['same', 'same', 'same', 'same', 'same', 'valid'],
        pooling_size=[2, 2, 2, 2, 2],
        pooling_type=['max_pool', 'max_pool', 'max_pool', 'max_pool', 'max_pool'],
        normalization='batch',
        dropout=False,
        #dropout_rate=0.5,
        softmax=False,
        use_float16=False,
        reduce_lr_on_plateau=0.5,
        batch_size=batch_size, 
        early_stopping=10,
        gpu_list = gpu_list,
        name=name+'_'+str(index),)

    generator_batch_size = model.get_batchsize()

    train_gen = VolumeDataGeneratorRegression(
        sample_df=train_df, 
        sample_stats_df=train_stats,
        batch_size=generator_batch_size, 
        #num_reg_classes=num_output, 
        dim=input_dim,
        input_preprocessing=input_preprocess,
        output_preprocessing='quantile', 
        idps_labels=idps_labels)

    scaler_instance = train_gen.get_scaler_instance()

    valid_gen = VolumeDataGeneratorRegression(
        sample_df=valid_df, 
        sample_stats_df=train_stats,
        batch_size=generator_batch_size, 
        #num_reg_classes=num_output, 
        dim=input_dim,
        input_preprocessing=input_preprocess,
        output_scaler=scaler_instance,
        shuffle=False, 
        idps_labels=idps_labels)

    
    if not only_evaluate:
        start = time.time()
        model.compile(learning_rate=1e-3, optimizer='Adam')
        model.train_generator(train_gen, valid_gen, epochs=epochs_num, workers=cpu_workers, verbose=2)

        time_elapsed = time.time() - start
        print('time elapsed (hours): {}'.format(time_elapsed/(3600)))


    model.load_weights('weights/checkpoint_' + name + '_' + str(index))

    test_df = pd.read_csv('csv/split_test.csv', index_col='id').dropna()

    test_gen = VolumeDataGeneratorRegression(
        sample_df=test_df, 
        sample_stats_df=train_stats,
        batch_size=generator_batch_size, 
        #num_reg_classes=num_output, 
        dim=input_dim,
        input_preprocessing=input_preprocess,
        output_scaler=scaler_instance, 
        idps_labels=idps_labels,
        shuffle=False
    )
    model.evaluate_generator(test_gen, filename=name + '_test', workers=cpu_workers)
    
    K.clear_session()

if __name__=='__main__':
    # for i in range(int(sys.argv[1])): 
    #     main(i)
    sfcn_pyramid_small_kern(sys.argv[1], only_evaluate=False)
    #train_and_evaluate(1, only_evaluate=True)
