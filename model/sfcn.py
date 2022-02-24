import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import Activation, Conv3D, MaxPooling3D, AveragePooling3D, BatchNormalization, LayerNormalization, Input, Dropout, Flatten,  Softmax, Reshape, UpSampling3D
from tensorflow.keras.models import Model
from model.custommodel import CustomTrainStep
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow_addons.layers import GroupNormalization, WeightNormalization
from tensorflow.distribute import MirroredStrategy

from model.conv3dwithws import Conv3DWithWeightStandardization

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import date
import os


class SFCN:
    def __init__(
        self,
        input_dim,
        output_dim,
        conv_num_filters,
        conv_kernel_sizes, 
        conv_strides,
        conv_padding, 
        pooling_type,
        pooling_size,
        activation='relu',
        normalization='batch',
        groupnorm_n = 4,
        weight_standardization=False,
        dropout=False,
        dropout_rate=0.5,
        softmax=False,
        global_pooling='avg_pool',
        input_resample='none',        
        batch_size=8,
        early_stopping=8, 
        reduce_lr_on_plateau=1,
        use_float16=False,
        gpu_list = range(8),
        name='SFCN'):
        """_summary_

        Parameters
        ----------
        input_dim : list of int
            dimension of input
        output_dim : int
            num of classes
        conv_num_filters : list of int
            number of filter for each layer. lenght determines depth
        conv_kernel_sizes : list of int
            kernel size. length should be the same as conv_num_filters
        conv_strides : list of int
            strides. length should be the same as conv_num_filters
        conv_padding : list or string
            choose 'same' or 'valid'.
        pooling_size : list of int
            pooling size. length should be the same as len(conv_num_filters)-1
        pooling_type : list or string
            pooling type. choose 'avg_pool', 'max_pool', or 'none' for not using pooling.
        activation : str, optional
            activation for each layer, by default 'relu'
        normalization : str, optional
            normalization for each layer. can be 'batch', 'layer' or 'group'. 'none' for not using normalization. by default 'batch'.
        groupnorm_n : int, optional
            number of group for group normalization, by default 4
        weight_standardization : bool, optional
            using weight normalization if True, by default False
        dropout : bool, optional
            add dropout at the end of the network, by default False
        dropout_rate : float, optional
            dropout rate it dropout is True, by default 0.5
        softmax : bool, optional
            use softmax. use only for classification. by default False
        global_pooling : str, optional
            pooling before last 1x1 convolution. choice are 'avg_pool' or 'max_pool'. by default 'avg_pool'
        input_resample : str, optional
            'downsample' or 'upsample' input. use downsample to reduce memory. by default 'none'
        batch_size : int, optional
            batchsize, by default 8
        early_stopping : int, optional
            num epochs before halting the training, by default 8
        reduce_lr_on_plateau : int, optional
            number of waiting epoch of, by default 1
        use_float16 : bool, optional
            use 16 bit floating point operation. saves GPU memory, by default False
        gpu_list : _type_, optional
            list of GPU to be used, by default range(8)
        name : str, optional
            network name, by default 'SFCN'
        """
        
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.conv_num_filters = conv_num_filters
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.conv_padding = conv_padding
        self.pooling_size = pooling_size
        self.pooling_type = pooling_type
        self.normalization = normalization
        self.groupnorm_n = groupnorm_n
        self.weight_standardization = weight_standardization
        self.activation = activation
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.softmax = softmax
        self.global_pooling = global_pooling
        self.input_resample = input_resample

        self.early_stopping = early_stopping
        self.reduce_lr_on_plateau = reduce_lr_on_plateau

        self.name = name

        self.n_conv_layer = len(conv_num_filters)

        self.num_gpu = len(gpu_list)     
        gpu_list_str = str()
        for gpu in gpu_list:
            gpu_list_str += str(gpu)+','
        gpu_list_str=gpu_list_str[:-1]

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list_str

        if use_float16:
            tf.keras.mixed_precision.set_global_policy('mixed_float16') 

        self.multi_gpu = self.num_gpu > 1

        self.batch_size = batch_size

        if self.multi_gpu:
            self.strategy = MirroredStrategy(tf.config.list_logical_devices('GPU'))
            with self.strategy.scope():
                self.build()
        else: 
            self.build()

    def build(self):
        """
        Build the SFCN Network
        """
        # Building model
        model_input = Input(shape=self.input_dim, name='input')

        x = model_input 

        if self.input_resample == 'downsample':
            x = MaxPooling3D(name='downsample_0')(x)
        elif self.input_resample == 'upsample':
            x = UpSampling3D(name='upsample_0')(x)

        for i in range(self.n_conv_layer-1):            
            if self.weight_standardization:
                x = Conv3DWithWeightStandardization(
                    filters=self.conv_num_filters[i],
                    kernel_size=self.conv_kernel_sizes[i],
                    strides=self.conv_strides[i],
                    padding=self.conv_padding[i],
                    name='conv_' + str(i))(x)
            else:
                x = Conv3D(
                    filters=self.conv_num_filters[i],
                    kernel_size=self.conv_kernel_sizes[i],
                    strides=self.conv_strides[i],
                    padding=self.conv_padding[i],
                    name='conv_' + str(i))(x)

            
            if self.normalization == 'batch':
                x = BatchNormalization(name='batchnorm_' + str(i))(x)
            elif self.normalization == 'layer':
                x = LayerNormalization(name='layernorm_' + str(i))(x)
            elif self.normalization == 'group':
                x = GroupNormalization(name='groupnorm_' + str(i), groups=self.groupnorm_n)(x)
            
            if self.pooling_type[i] == 'avg_pool':
                x = AveragePooling3D(pool_size=self.pooling_size[i], name='avgpool_' + str(i))(x)
            elif self.pooling_type[i] == 'max_pool':
                x = MaxPooling3D(pool_size=self.pooling_size[i], name='maxpool_' + str(i))(x)
            
            x = Activation(self.activation, name='activation_' + str(i))(x)

        if self.weight_standardization:
            x = Conv3DWithWeightStandardization(
                filters=self.conv_num_filters[i],
                kernel_size=self.conv_kernel_sizes[i],
                strides=self.conv_strides[i],
                padding=self.conv_padding[i],
                name='conv_' + str(self.n_conv_layer-1))(x)
        else:
            x = Conv3D(
                filters=self.conv_num_filters[i],
                kernel_size=self.conv_kernel_sizes[i],
                strides=self.conv_strides[i],
                padding=self.conv_padding[i],
                name='conv_' + str(self.n_conv_layer-1))(x)

        if self.normalization == 'batch':
            x = BatchNormalization(name='batchnorm_' + str(self.n_conv_layer-1))(x)
        elif self.normalization == 'layer':
            x = LayerNormalization(name='layernorm_' + str(self.n_conv_layer-1))(x)
        elif self.normalization == 'group':
            x = GroupNormalization(name='groupnorm_' + str(self.n_conv_layer-1), groups=self.groupnorm_n)(x)

        x = Activation(self.activation, name='activation_' + str(self.n_conv_layer-1))(x)

        avg_shape = x.shape.as_list()
        if self.global_pooling == 'avg_pool':
            x = AveragePooling3D(pool_size=avg_shape[1:-1], name='avgpool_'+ str(self.n_conv_layer))(x)
        elif self.global_pooling == 'max_pool':
            x = MaxPooling3D(pool_size=avg_shape[1:-1], name='maxpool_'+ str(self.n_conv_layer))(x)

        if self.dropout:
            x = Dropout(rate=self.dropout_rate, name='dropout_'+str(self.n_conv_layer))(x)

        x = Conv3D(filters=self.output_dim, kernel_size=1, name='conv_'+str(self.n_conv_layer))(x)
        x = Flatten()(x)
        
        if self.softmax:
            x = Softmax()(x)
        
        model_output = x
        
        self.model = Model(model_input, model_output, name=self.name)
        
        self.model.summary()
                
        # building callbacks
        checkpoint_filepath = 'weights/checkpoint_'+self.name

        self.callbacks = [ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True)]

        if self.early_stopping > 0:
            self.callbacks.append(EarlyStopping(patience=self.early_stopping))

        if self.reduce_lr_on_plateau < 1:
            self.callbacks.append(ReduceLROnPlateau(patience=1, factor=self.reduce_lr_on_plateau))


    def compile(self, learning_rate=1e-6, optimizer='Adam', loss = 'mse'):
        """compile the model. call when starting training

        Parameters
        ----------
        learning_rate : float, optional
            learning rate of the optimizer, by default 1e-6
        optimizer : str, optional
            can be Adam or SGD, by default 'Adam'
        loss : str, optional
            loss function, by default 'mse'
        """
        if optimizer=='Adam':
            self.optimizer = Adam(learning_rate=learning_rate)
        elif optimizer=='SGD':
            self.optimizer = SGD(learning_rate=learning_rate)
    
        if self.multi_gpu:
            with self.strategy.scope():            
                    self.model.compile(optimizer=self.optimizer, loss=loss, metrics=['mae'])       
        else:           
            self.model.compile(optimizer=self.optimizer, loss=loss, metrics=['mae'])


    def train_generator(self, train_generator, valid_generator, epochs, workers=4, queue_size=None, verbose=2):
        """training the model with volumedatagenerator

        Parameters
        ----------
        train_generator : volumedatagenerator
            training data generator 
        valid_generator : volumedatagenerator
            validation data generator
        epochs : int
            number of epochs
        workers : int, optional
            number of cpu workers for the generator. pick a bigger number if GPU is mostly idle. by default 4
        queue_size : int, optional
            data queue size. pick a bigger number if GPU is mostly idle. by default None
        verbose : int, optional
            training status. pick 1 for silent, 2 for more detailed, by default 2
        """
        if queue_size==None:
           queue_size=workers

        self.history = self.model.fit(
            x = train_generator, 
            validation_data = valid_generator, 
            batch_size = self.batch_size, 
            epochs = epochs, 
            callbacks = self.callbacks,
            workers = workers,
            max_queue_size = queue_size,
            verbose = verbose)
                
        self.save_history()   


    def load_weights(self, filepath):
        """loading weights into the model

        Parameters
        ----------
        filepath : str
            weights path
        """
        if self.multi_gpu:
            with self.strategy.scope():
                self.model.load_weights(filepath=filepath)
        else:
            self.model.load_weights(filepath=filepath)


    def evaluate_generator(self, x_generator, filename=None, workers=4, queue_size=None):
        """evaluate the model by feeding test set volumegenerator

        Parameters
        ----------
        x_generator : volumedatagenerator
            test set generator
        filename : str, optional
            file name for result file, if none, using self.name. by default None
        workers : int, optional
            cpu worker for the generator, by default 4
        queue_size : int, optional
            queue size for the generator, by default None
        """
        if queue_size==None:
           queue_size=workers
        
        y_pred = self.model.predict(
            x = x_generator,
            batch_size=self.batch_size, 
            workers=workers,
            max_queue_size=queue_size)

        # aggregated r2 score
        y_true = x_generator.get_labels()[:y_pred.shape[0],:] # Batching leaves some samples out, so we should throw y_true labels out too

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        today = date.today()

        if filename==None:
            filename=self.name

        filedir = Path.cwd().joinpath('results')
        filedir.mkdir(parents=True, exist_ok=True)
        fn = filedir.joinpath(filename +'.csv')

        if fn.exists():
            entry ={'date':today, 'r2':r2, 'mae':mae, 'mse':mse }
            df = pd.read_csv(fn)    
            df = df.append(entry, ignore_index=True)
        else:
            columns = ['date','r2','mae','mse']
            df = pd.DataFrame([[today, r2, mae, mse]], columns=columns)

        df.to_csv(fn, index=False)

        # individual r2 and mse scores
        num_class = y_true.shape[1]
                
        multi_r2 = list()
        multi_mse = list()
        for i in range(num_class):
            multi_r2.append(r2_score(y_true[:,i], y_pred[:,i]))
            multi_mse.append(mean_squared_error(y_true[:,i], y_pred[:,i]))

        labels = x_generator.get_column_labels()
                  
        #R2
        r2fn = filedir.joinpath(filename + '_multi_r2.csv')
        if r2fn.exists():
            entry = dict(zip(labels, multi_r2))
            df = pd.read_csv(r2fn)
            df = df.append(entry, ignore_index=True)
        else: 
            df = pd.DataFrame([multi_r2], columns=labels)
        df.to_csv(r2fn, index=False)

        #MSE
        msefn = filedir.joinpath(filename + '_multi_mse.csv')
        if msefn.exists():
            entry = dict(zip(labels, multi_mse))
            df = pd.read_csv(msefn)
            df = df.append(entry, ignore_index=True)
        else: 
            df = pd.DataFrame([multi_mse], columns=labels)
        df.to_csv(msefn, index=False)


    def get_history(self):
        return self.history


    def save_history(self):
        filedir = Path.cwd().joinpath('history')
        filedir.mkdir(parents=True, exist_ok=True)
        fn = filedir.joinpath(self.name+'_hist.csv')

        df = pd.DataFrame(self.history.history)
        df.to_csv(fn, index=False)

    def get_batchsize(self):
        return self.batch_size