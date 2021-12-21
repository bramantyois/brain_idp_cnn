import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import Activation, Conv3D, MaxPooling3D, AveragePooling3D, BatchNormalization, Input, Dropout, Flatten, Dense, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.distribute import MirroredStrategy

import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import os 
from pathlib import Path
from datetime import date


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
        dropout_rate=0.5,
        softmax=False,
        gpu_num = 2,
        use_float16=False,
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
        self.name = name

        self.n_conv_layer = len(conv_num_filters)

        if use_float16:
            tf.keras.mixed_precision.set_global_policy("mixed_float16") 

        gpus = tf.config.list_logical_devices('GPU')[:gpu_num]        
        self.strategy = MirroredStrategy(gpus)
        
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
            else:
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
        x = AveragePooling3D(pool_size=avg_shape, name='avgpool_1'+ str(self.n_conv_layer-1))(x)

        if self.dropout:
            x = Dropout(rate=self.dropout_rate)(x)

        x = Conv3D(filters=self.output_dim, kernel_size=1, name='conv_'+str(self.n_conv_layer + 1))(x)
        
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


    def train_generator(self, train_generator, valid_generator, batch_size, epochs, workers=4, queue_size=32):
        """[summary]

        Args:
            train_generator ([type]): [description]
            valid_generator ([type]): [description]
            batch_size ([type]): [description]
            epochs ([type]): [description]
            workers (int, optional): [description]. Defaults to 4.
        """

        self.history = self.model.fit(
            x=train_generator, 
            validation_data=valid_generator, 
            batch_size=batch_size, 
            epochs=epochs, 
            callbacks=self.callbacks,
            workers=workers,
            max_queue_size=queue_size)
        
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

    def evaluate_generator(self, x_generator, batch_size, filename=None, workers=4, queue_size=32):
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


    def get_history(self):
        return self.history

    def save_history(self):
        filedir = Path.cwd().joinpath('history')
        filedir.mkdir(parents=True, exist_ok=True)
        fn = filedir.joinpath(self.name+'_hist.csv')

        df = pd.DataFrame(self.history.history)
        df.to_csv(fn, index=False)





        