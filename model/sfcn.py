import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import Activation, Conv3D, MaxPooling3D, AveragePooling3D, BatchNormalization, Input, Dropout, Flatten, Dense, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.distribute import MirroredStrategy
from tensorflow_addons.optimizers import extend_with_decoupled_weight_decay
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import date
import os

class SFCN():
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
        batch_norm=True,
        dropout=True,
        dropout_rate=0.9,
        softmax=False,
        use_float16=False,
        early_stopping=8, 
        reduce_lr_on_plateau=1,
        gpu_list = range(8),
        name='SFCN'):
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
            [description], by default 0.9
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

        self.early_stopping = early_stopping
        self.reduce_lr_on_plateau = reduce_lr_on_plateau

        self.name = name

        self.n_conv_layer = len(conv_num_filters)

        gpu_list_str = str()
        for gpu in gpu_list:
            gpu_list_str += str(gpu)+','
        gpu_list_str=gpu_list_str[:-1]

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list_str

        gpu_num = len(gpu_list)
        gpus = tf.config.list_logical_devices('GPU')        
                
        if use_float16:
            tf.keras.mixed_precision.set_global_policy("mixed_float16") 

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
        model_input = Input(shape=self.input_dim, name='input')
        
        x = model_input 

        for i in range(self.n_conv_layer-1):
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
            elif self.pooling_type[i] == 'max_pool':
                x = MaxPooling3D(pool_size=self.pooling_size[i], name='maxpool_' + str(i))(x)
            
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
        """[summary]

        Args:
            learning_rate ([type]): [description]
            optimizer (str, optional): [description]. Defaults to 'adam'.
            loss (str, optional): [description]. Defaults to 'mse'.
        """
        
        with self.strategy.scope():
            self.learning_rate = learning_rate
            if optimizer=='Adam':
                self.optimizer = Adam(learning_rate=learning_rate)
            elif optimizer=='SGD':
                self.optimizer = SGD(learning_rate=learning_rate)
            elif optimizer=='AdamW':
                MyAdamW = extend_with_decoupled_weight_decay(Adam)
                self.optimizer = MyAdamW(learning_rate=learning_rate, weight_decay=0.001)


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





        