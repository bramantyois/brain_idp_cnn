from typing import Tuple
import nibabel as nib

import numpy as np
import pandas as pd

import math
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer


class VolumeDataGeneratorRegression(Sequence):
    def __init__(
        self, 
        sample_df, 
        sample_stats_df,
        batch_size=8, 
        shuffle=True, 
        dim=(160, 182, 160), 
        num_channels=1,
        input_preprocessing='none',
        output_preprocessing='none',
        output_scaler=None,
        idps_labels = list()):
        """Data Generator for three dimensional data for regression purposes

        Parameters
        ----------
        sample_df : pd.DataFrame
            dataframe containing path of .nii.gz file and its descriptors
        sample_stats_df : pd.DataFrame
            dataframe containing statistics of images correspond to sample_df
        batch_size : int, optional
            batch size, by default 8
        shuffle : bool, optional
            shuffle the order if True, by default True
        dim : tuple, optional
            dimension of input. if smaller than the actual data, will crop. by default (160, 182, 160)
        num_channels : int, optional
            num of channels, by default 1
        input_preprocessing : str, optional
            preprocessing of input. 'standardize' of standardization, 'normalize' for normaliztion. by default 'none'
        output_preprocessing : str, optional
            output label preprocessing.chooses 'standard', 'minmax','quantile' or 'quantile-normal' by default 'none'
        output_scaler : sklearn scaler instance, optional
            scaler instance. use this when having external instances e.g scaler fitted on training data. override output_preprocessing. by default None
        idps_labels : list, optional
            list of idps labels to be trained for. lenghts determines output size, by default list()
        """
        #sample_df = sample_df.copy().dropna()

        self.sample_df = sample_df['path']
        self.sample_stats_df = sample_stats_df

        # if not self.sample_df.index.equals(self.sample_stats_df.index):
        #     print('sample_df and stats are not the same')

        self.target_df = sample_df.drop('path',axis=1)

        if len(idps_labels) !=0:
            self.target_df = self.target_df[idps_labels]
        
        self.column_label_names = self.target_df.columns.to_list()
        # print(self.target_df.shape)

        self.num_data = len(sample_df.index)
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = dim
        self.num_channels = num_channels
        self.num_classes = len(self.target_df.columns)

        self.input_preprocessing = input_preprocessing

        self.output_preprocessing = output_preprocessing
        self.output_scaler_inst = output_scaler
        
        #self.compute_input_preprocessing_param()
        self.preprocess_output()
        self.on_epoch_end()


    def preprocess_output(self):
        """output label preprocessing
        """
        if self.output_scaler_inst==None:
            if self.output_preprocessing == 'standard':
                self.output_scaler_inst = StandardScaler()
                
            elif self.output_preprocessing == 'minmax':
                self.output_scaler_inst = MinMaxScaler()

            elif self.output_preprocessing =='quantile':
                self.output_scaler_inst = QuantileTransformer(n_quantiles=1000)        
                        
            elif self.output_preprocessing =='quantile-normal':
                self.output_scaler_inst = QuantileTransformer(n_quantiles=1000, output_distribution='normal')   

            if self.output_preprocessing in ['standard', 'minmax', 'quantile', 'quantile-normal']:
                transformed  = self.output_scaler_inst.fit_transform(self.target_df)
                self.target_df = pd.DataFrame(transformed, index=self.target_df.index, columns=self.column_label_names)
            
        else:
            transformed  = self.output_scaler_inst.transform(self.target_df)
            self.target_df = pd.DataFrame(transformed, index=self.target_df.index, columns=self.column_label_names)
        
        if (self.target_df.isnull().values.any()):
            print('nan target found')
            # to do remove nan entries
    

    def get_scaler_instance(self):
        """return scaler instance

        Returns
        -------
        sklearn scaler instance
            _description_
        """
        return self.output_scaler_inst


    def get_labels(self):
        """return the whole labels of the generator
        """
        return self.target_df.iloc[self.indices].to_numpy()

    def get_column_labels(self):
        """return column labels as a list
        """
        return self.column_label_names

    def on_epoch_end(self):
        """
        Shuffle the indices when shuffle==True. otherwise keep the index order
        """
        self.indices = np.arange(self.num_data)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def crop(self, input):
        """croping input so that it has size of self.input_dim

        Parameters
        ----------
        input : numpy volume
            input volume

        Returns
        -------
        numpy volume
            cropped input
        """
        output_shape = self.dim
        input_shape = input.shape

        x_crop = int(0.5*(input_shape[0] - output_shape[0]))
        y_crop = int(0.5*(input_shape[1] - output_shape[1]))
        z_crop = int(0.5*(input_shape[2] - output_shape[2]))
        return input[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop] 

    def __len__(self):
        """
        Return the number of batches per epoch
        """
        return math.floor(self.num_data/self.batch_size) 
    
    
    def __getitem__(self, index) -> Tuple:
        """Get one batch of data

        Parameters
        ----------
        index : int
            Batch index

        Returns
        -------
        Tuple
            Tuple of data and labels with size of batch_size
        """
        indices = self.indices[index * self.batch_size: (index+1)*self.batch_size]     
        
        # paths = [self.sample_df.iloc[i] for i in indices]

        # initialize arrays for volumes and labels
        X = np.zeros((self.batch_size, *self.dim, self.num_channels), dtype=np.float32)
        Y = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)

        if self.input_preprocessing == 'none':
            for i in range(0, self.batch_size):            
                idx = indices[i]     
                img = self.crop(nib.load(self.sample_df.iloc[idx]).get_fdata())
                X[i] = img.reshape((*self.dim,1))
                Y[i] = self.target_df.iloc[idx].to_numpy()

        elif self.input_preprocessing == 'standardize':
            for i in range(self.batch_size):            
                idx = indices[i]    

                img = self.crop(nib.load(self.sample_df.iloc[idx]).get_fdata())

                img = (img - self.sample_stats_df.iloc[0]['mean']) / self.sample_stats_df.iloc[0]['std']

                X[i] = img.reshape((*self.dim,1))
                Y[i] = self.target_df.iloc[idx].to_numpy()

        elif self.input_preprocessing == 'normalize':
            for i in range(self.batch_size):  
                idx = indices[i]    

                img = self.crop(nib.load(self.sample_df.iloc[idx]).get_fdata())

                img = (img - self.sample_stats_df.iloc[0]['min']) / self.sample_stats_df.iloc[0]['range']

                X[i] = img.reshape((*self.dim,1))
                Y[i] = self.target_df.iloc[idx].to_numpy()

        return X, Y
            










