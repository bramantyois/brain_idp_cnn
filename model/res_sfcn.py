import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import Add, Activation, Conv3D, MaxPooling3D, AveragePooling3D, BatchNormalization, Input, Dropout, Flatten, Dense, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.distribute import MirroredStrategy

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import date
import os


def residual_block(
    x, 
    filter_num, 
    kernel_size=2, 
    padding = 'same',
    downsample=True, 
    prefix='res', 
    postfix='0'):
    """[summary]

    Args:
        x (Keras Tensor): Input Tensor
        filter_num (int): filter num
        kernel_size (int, optional): kernel size. Defaults to 3.
        padding (String, optional): Padding, better leave it 'same'. Defaults to 'same'.
        downsample (bool, optional): downsampling . Defaults to True.
        downsample_avgpool (bool, optional): pooling applied to x before conv if true. Defaults to True.
        prefix (str, optional): name prefix . Defaults to 'res'.
        postfix (str, optional): name postfix. Defaults to '0'.

    Returns:
        Keras Tensor: result
    """

    y = Conv3D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filter_num,
               padding=padding, 
               activation='relu',
               name = prefix + '_conv0_' + postfix)(x)

    y = BatchNormalization(name = prefix + '_bn0_' + postfix)(y)

    y = Conv3D(kernel_size=kernel_size,
               strides=1,
               filters=filter_num,
               padding=padding,
               name = prefix + '_conv1_' + postfix)(y)

    if downsample:
        ds_stride=2     
        x = Conv3D(kernel_size=1,
                strides=ds_stride,
                filters=filter_num,
                padding="same",
                name = prefix + '_ds_conv_' + postfix)(x)

    out = Add(name = prefix + '_add_' + postfix)([x, y])
    out = Activation('relu', name = prefix + '_relu_' + postfix)(out)
    out = BatchNormalization(name = prefix + '_bn1_' + postfix)(out)

    return out


class ResSFCN():
    def __init__(
        self,
        input_dim,
        output_dim,
        conv_num_filters,
        conv_kernel_sizes, 
        conv_strides,
        conv_padding, 
        # pooling_size,
        # pooling_type,
        normalization='batch',
        dropout=False,
        dropout_rate=0.5,
        softmax=False,
        global_pooling='avg_pool',
        batch_size=8,
        early_stopping=16, 
        reduce_lr_on_plateau=1,
        use_float16=True,
        gpu_list = range(8),
        name='ResSFCN'):
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
        # self.pooling_size = pooling_size
        # self.pooling_type = pooling_type
        self.normalization = normalization
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.softmax = softmax
        self.global_pooling = global_pooling

        self.name = name

        self.early_stopping = early_stopping
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        
        self.n_conv_layer = len(conv_num_filters)

        gpu_list_str = str()
        for gpu in gpu_list:
            gpu_list_str += str(gpu)+','
        gpu_list_str=gpu_list_str[:-1]

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list_str

        if use_float16:
            tf.keras.mixed_precision.set_global_policy("mixed_float16") 

        self.num_gpu = len(gpu_list)     
        self.multi_gpu = self.num_gpu > 1

        self.batch_size = batch_size
        self.num_accum = 1

        if self.multi_gpu:
            self.strategy = MirroredStrategy(tf.config.list_logical_devices('GPU'))
            with self.strategy.scope():
                self.build()
        else: 
            self.build()
    

    def build(self):
        """[summary]
        """
        # Building model

        model_input = Input(shape=self.input_dim, name='input')

        x = model_input 

        for i in range(self.n_conv_layer-1):
            x = residual_block(
                x=x,
                filter_num=self.conv_num_filters[i],
                kernel_size=self.conv_kernel_sizes[i],
                downsample=True, 
                postfix=str(i)
            )

        x = residual_block(
            x=x,
            filter_num=self.conv_num_filters[-1],
            kernel_size=self.conv_kernel_sizes[-1],
            downsample=False, 
            postfix=str(self.n_conv_layer-1))

      
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


    def train_generator(self, train_generator, valid_generator, epochs, workers=4, queue_size=None, verbose=2):
        """[summary]

        Args:
            train_generator ([type]): [description]
            valid_generator ([type]): [description]
            batch_size ([type]): [description]
            epochs ([type]): [description]
            workers (int, optional): [description]. Defaults to 4.
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
        """[summary]

        Parameters
        ----------
        filepath : [type]
            [description]
        """
        if self.multi_gpu:
            with self.strategy.scope():
                self.model.load_weights(filepath=filepath)
        else:
            self.model.load_weights(filepath=filepath)


    def predict(self, x):
        return self.model.predict(x)


    def evaluate_generator(self, x_generator, filename=None, workers=4, queue_size=None):
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
