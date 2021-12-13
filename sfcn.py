import tensorflow as tf
from tensorflow.distribute import MirroredStrategy

from tensorflow import keras 
from tensorflow.keras.layers import Activation, Conv3D, MaxPooling3D, AveragePooling3D, BatchNormalization, Input, Dropout, Flatten, Dense, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

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
        maxpool_size,
        batch_norm=True,
        dropout=True,
        dropout_rate=0.5,
        softmax=False,
        gpu_num = 2,
        name='SFCN'):
        """[summary]

        Args:
            input_dim ([type]): [description]
            conv_num_filters ([type]): [description]
            conv_kernel_sizes ([type]): [description]
            conv_strides ([type]): [description]
            conv_padding ([type]): [description]
            maxpool_size ([type]): [description]
            batch_norm (bool, optional): [description]. Defaults to True.
            dropout (bool, optional): [description]. Defaults to True.
            dropout_rate (float, optional): [description]. Defaults to 0.5.
            softmax (bool, optional): [description]. Defaults to True.
        """
        
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.conv_num_filters = conv_num_filters
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.conv_padding = conv_padding
        self.maxpool_size = maxpool_size
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.softmax = softmax

        self.name = name

        self.n_conv_layer = len(conv_num_filters)

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

            x = MaxPooling3D(pool_size=self.maxpool_size[i], name='maxpool_' + str(i))(x)

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

        avg_shape = [5, 6, 5] # ! this should not be hardcoded
        x = AveragePooling3D(pool_size=avg_shape, name='avgpool_1')(x)

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
            EarlyStopping(patience=3),
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

        self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, callbacks=self.callbacks, validation_split=0.4)


    def train_generator(self, train_generator, valid_generator, batch_size, epochs, workers=4):
        """[summary]

        Args:
            train_generator ([type]): [description]
            valid_generator ([type]): [description]
            batch_size ([type]): [description]
            epochs ([type]): [description]
            workers (int, optional): [description]. Defaults to 4.
        """

        self.model.fit(
            x=train_generator, 
            validation_data=valid_generator, 
            batch_size=batch_size, 
            epochs=epochs, 
            callbacks=self.callbacks,
            workers=workers)


    def load_weights(self, filepath):
        with self.strategy.scope():
            #self.build()
            self.model.load_weights(filepath=filepath)


    def predict(self, x):
        return self.model.predict(x)