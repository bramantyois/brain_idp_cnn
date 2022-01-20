from model.sfcn import SFCN
import numpy as np
import pandas as pd
from volumedatagenerator import VolumeDataGeneratorRegression
import matplotlib.pyplot as plt

import sys
import time


def train_and_evaluate(idx, only_evaluate=False):
    name = 'sfcn_vanilla'
    index=int(idx)

    batch_size = 16
    generator_batch_size = 1
    #accum_num = 16
    #gpu_list = range(8)
    gpu_list = [7]
    cpu_workers = 8
    epochs_num = 64
    input_preprocess = 'standardize'

    idps_labels = pd.read_csv('csv/idps_desc.csv')['id'].to_list()
    idps_labels = [str(l) for l in idps_labels]

    train_df = pd.read_csv('csv/split_train.csv', index_col='id').dropna()
    train_stats = pd.read_csv('csv/train_stats.csv', index_col='id')

    valid_df = pd.read_csv('csv/split_valid.csv', index_col='id').dropna()
    valid_stats = pd.read_csv('csv/valid_stats.csv', index_col='id')

    input_dim = [160, 192, 160]
    num_output = len(idps_labels)

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
        sample_stats_df=valid_stats,
        batch_size=generator_batch_size, 
        #num_reg_classes=num_output, 
        dim=input_dim,
        input_preprocessing=input_preprocess,
        output_scaler=scaler_instance,
        shuffle=False, 
        idps_labels=idps_labels)

    model = SFCN(
        input_dim=[160, 192, 160, 1], 
        output_dim=num_output,
        conv_num_filters=[32, 64, 128, 256, 256, 64], 
        conv_kernel_sizes=[3, 3, 3, 3, 3, 1], 
        conv_strides=[1, 1, 1, 1, 1, 1],
        conv_padding=['same', 'same', 'same', 'same', 'same', 'valid'],
        pooling_size=[2, 2, 2, 2, 2],
        pooling_type=['max_pool', 'max_pool', 'max_pool', 'max_pool', 'max_pool'],
        normalization='layer',
        dropout=False,
        softmax=False,
        use_float16=False,
        reduce_lr_on_plateau=0.5,
        batch_size=batch_size, 
        early_stopping=16,
        gpu_list = gpu_list,
        name=name+'_'+str(index),)

    if not only_evaluate:
        start = time.time()
        model.compile(learning_rate=1e-3, optimizer='Adam')
        model.train_generator(train_gen, valid_gen, epochs=epochs_num, workers=cpu_workers, verbose=2)

        time_elapsed = time.time() - start
        print('time elapsed (hours): {}'.format(time_elapsed/(3600)))

    model.load_weights('weights/checkpoint_' + name + '_' + str(index))

    test_df = pd.read_csv('csv/split_test.csv', index_col='id').dropna()
    test_stats = pd.read_csv('csv/test_stats.csv', index_col='id')

    test_gen = VolumeDataGeneratorRegression(
        sample_df=test_df, 
        sample_stats_df=test_stats,
        batch_size=generator_batch_size, 
        #num_reg_classes=num_output, 
        dim=input_dim,
        input_preprocessing=input_preprocess,
        output_scaler=scaler_instance, 
        idps_labels=idps_labels,
        shuffle=False
    )
    model.evaluate_generator(test_gen, filename=name + '_test', workers=cpu_workers)


if __name__=='__main__':
    # for i in range(int(sys.argv[1])): 
    #     main(i)
    # train_and_evaluate(sys.argv[1])
    train_and_evaluate(13)