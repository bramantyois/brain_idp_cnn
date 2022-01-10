from re import S
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import Add, Activation, Conv3D, MaxPooling3D, AveragePooling3D, BatchNormalization, Input, Dropout, Flatten, Dense, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.distribute import MirroredStrategy

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import date
import math


def get_residuals_pool_shape(input_shape, output_shape):
    shape = list()

    for i in range(len(input_shape)):
        x = int(math.floor(input_shape[i]/output_shape[i]))
        shape.append(x)

    return x


def is_odd(val):
    if val%2==0:
        return False
    else:
        return True


class ResSFCN():
    def __init__(
        self,
        input_dim,
        output_dim,
        conv_num_filters,
        conv_kernel_sizes, 
        conv_strides,
        conv_padding, 
        pooling_size,
        pooling_type,
        res_pooling=True,
        batch_norm=True,
        dropout=True,
        dropout_rate=0.5,
        softmax=False,
        gpu_num = 2,
        use_float16=False,
        name='RESSFCN'):
        """[summary]

        Parameters
        ----------
        input_dim : [type]
            [description]
        output_dim : [type]
            [description]
        conv_num_filters : [type]
            [description]
        conv_kernel_sizes : [type]
            [description]
        conv_strides : [type]
            [description]
        conv_padding : [type]
            [description]
        pooling_size : [type]
            [description]
        pooling_type : [type]
            [description]
        batch_norm : bool, optional
            [description], by default True
        dropout : bool, optional
            [description], by default True
        dropout_rate : float, optional
            [description], by default 0.5
        softmax : bool, optional
            [description], by default False
        gpu_num : int, optional
            [description], by default 2
        name : str, optional
            [description], by default 'SFCN'
        """
        
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.conv_num_filters = conv_num_filters
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.conv_padding = conv_padding
        self.pooling_size = pooling_size
        self.pooling_type = pooling_type
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.softmax = softmax
        self.res_pooling = res_pooling
        self.name = name


        self.n_conv_layer = len(conv_num_filters)

        if use_float16:
            tf.keras.mixed_precision.set_global_policy("mixed_float16") 

        gpus = tf.config.list_logical_devices('GPU')[:gpu_num]        
        self.strategy = MirroredStrategy(gpus)
        with self.strategy.scope():
            self.build()
        #os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_index)
        #with tf.device("gpu:"+str(gpu_index)):
        #    print("tf.keras will run on GPU: {}".format(gpu_index))
    

    def build(self):
        """[summary]
        """
        # Building model
        input_copy = list()

        model_input = Input(shape=self.input_dim, name='input')

        x = model_input 

        n_half = int(math.floor(0.5*(self.n_conv_layer-1)))
    
        # first half convolutions
        for i in range(n_half):
            x = Conv3D(
                filters=self.conv_num_filters[i],
                kernel_size=self.conv_kernel_sizes[i],
                strides=self.conv_strides[i],
                padding=self.conv_padding[i],
                name='conv_' + str(i)
                )(x)
            
            if self.batch_norm:
                x = BatchNormalization(name='batchnorm_' + str(i))(x)

            if self.pooling_type[i] == 'avg_pool':
                x = AveragePooling3D(pool_size=self.pooling_size[i], name='avgpool_' + str(i))(x)
            else:
                x = MaxPooling3D(pool_size=self.pooling_size[i], name='maxpool_' + str(i))(x)

            input_copy.append(x)

            x = Activation('relu', name='activation_' + str(i))(x)
   
        # last half convolutions
        for i in range(n_half, self.n_conv_layer-1):
            x = Conv3D(
                filters=self.conv_num_filters[i],
                kernel_size=self.conv_kernel_sizes[i],
                strides=self.conv_strides[i],
                padding=self.conv_padding[i],
                name='conv_' + str(i)
                )(x)
            
            if self.batch_norm:
                x = BatchNormalization(name='batchnorm_' + str(i))(x)

            if self.pooling_type[i] == 'avg_pool':
                x = AveragePooling3D(pool_size=self.pooling_size[i], name='avgpool_' + str(i))(x)
            else:
                x = MaxPooling3D(pool_size=self.pooling_size[i], name='maxpool_' + str(i))(x)

            res_idx =  self.n_conv_layer-i-1 
            if (res_idx <= n_half):
                #now adding residuals
                res = input_copy[n_half-1-res_idx]

                cur_shape = x.shape.as_list()[1:-1]
                res_shape = res.shape.as_list()[1:-1]

                pool_shape = get_residuals_pool_shape(res_shape, cur_shape)

                res_num_fil = x.shape.as_list()[-1]
                if self.res_pooling:
                    res = AveragePooling3D(pool_size=pool_shape, name='res_avgpool_'+str(i))(res)
                    res = Conv3D(filters=res_num_fil, kernel_size=1,  name='res_conv_'+str(i))(res)
                else:
                    # to be implemented
                    res = AveragePooling3D(pool_size=pool_shape, name='res_avgpool_'+str(i))(res)
                    res = Conv3D(filters=res_num_fil, kernel_size=1,  name='res_conv_'+str(i))(res)
                    # res = Conv3D(filters=res_num_fil, kernel_size=1, strides=pool_shape, padding='valid', name='res_conv_'+str(i))(res)

                x = Add()([x, res])
                
            # relu at the end of the block
            x = Activation('relu', name='activation_' + str(i))(x)

        x = Conv3D(
            filters=self.conv_num_filters[-1],
            kernel_size=self.conv_kernel_sizes[-1],
            strides=self.conv_strides[-1],
            padding=self.conv_padding[-1],
            name='conv_' + str(self.n_conv_layer-1)
            )(x)

        if self.batch_norm:
            x = BatchNormalization(name='batchnorm_' + str(self.n_conv_layer-1))(x)
       
        x = Activation('relu', name='activation_' + str(self.n_conv_layer-1))(x)

        avg_shape = x.shape.as_list()[1:-1]
        x = AveragePooling3D(pool_size=avg_shape, name='avgpool_'+ str(self.n_conv_layer))(x)

        if self.dropout:
            x = Dropout(rate=self.dropout_rate)(x)

        x = Conv3D(filters=self.output_dim, kernel_size=1, name='conv_'+str(self.n_conv_layer))(x)
        
        x = Flatten()(x)
        if self.softmax:
            x = Softmax()(x)
        
        model_output = x
        # x = Flatten()(x)
        # model_output = Dense(units=self.output_dim)(x)
        
        self.model = Model(model_input, model_output)
        self.model.summary()
        
        
        # building callbacks
        checkpoint_filepath = 'weights/checkpoint_'+self.name
        self.callbacks =  [
            EarlyStopping(patience=16),
            ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_mae',
                mode='min',
                save_best_only=True)]


    def compile(self, learning_rate, optimizer='Adam', loss = 'mse'):
        """[summary]

        Args:
            learning_rate ([type]): [description]
            optimizer (str, optional): [description]. Defaults to 'adam'.
            loss (str, optional): [description]. Defaults to 'mse'.
        """
        
        with self.strategy.scope():
            self.build()

            self.learning_rate = learning_rate
            if optimizer=='Adam':
                self.optimizer = Adam(learning_rate=learning_rate)
            elif optimizer=='SGD':
                self.optimizer = SGD(learning_rate=learning_rate)

            if loss == 'mse':
                self.model.compile(optimizer=self.optimizer, loss=loss, metrics=['mae'])

    
    def train(self, x_train, y_train, batch_size, epochs):
        """[summary]

        Args:
            x_train ([type]): [description]
            y_train ([type]): [description]
            batch_size ([type]): [description]
            epochs ([type]): [description]
        """

        self.history = self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, callbacks=self.callbacks, validation_split=0.4)

        self.save_history()


    def train_generator(self, train_generator, valid_generator, batch_size, epochs, workers=4, queue_size=None):
        """[summary]

        Args:
            train_generator ([type]): [description]
            valid_generator ([type]): [description]
            batch_size ([type]): [description]
            epochs ([type]): [description]
            workers (int, optional): [description]. Defaults to 4.
        """
        if queue_size==None:
            queue_size=batch_size

        self.history = self.model.fit(
            x=train_generator, 
            validation_data=valid_generator, 
            batch_size=batch_size, 
            epochs=epochs, 
            callbacks=self.callbacks,
            workers=workers,
            max_queue_size=queue_size,
            verbose=2)
        
        self.save_history()   


    def load_weights(self, filepath):
        """[summary]

        Parameters
        ----------
        filepath : [type]
            [description]
        """
        with self.strategy.scope():
            self.model.load_weights(filepath=filepath)


    def predict(self, x):
        return self.model.predict(x)

    def evaluate_generator(self, x_generator, batch_size, filename=None, workers=4, queue_size=None):
        if queue_size==None:
            queue_size=batch_size

        y_pred = self.model.predict(
            x = x_generator,
            batch_size=batch_size, 
            workers=workers,
            max_queue_size=queue_size)

        # Batching leaves some samples out, so we should throw y_true labels out too
        y_true = x_generator.get_labels()[:y_pred.shape[0],:]

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

        