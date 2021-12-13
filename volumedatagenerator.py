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
        batch_size=16, 
        shuffle=True, 
        dim=(160, 160, 91), 
        num_channels=1,
        num_reg_classes=1,
        output_preprocessing='none',
        output_scaler=None,
        verbose=False):
        """[summary]

        Parameters
        ----------
        sample_df : [type]
            [description]
        batch_size : int, optional
            [description], by default 16
        shuffle : bool, optional
            [description], by default True
        dim : tuple, optional
            [description], by default (160, 160, 91)
        num_channels : int, optional
            [description], by default 1
        num_reg_classes : int, optional
            [description], by default 1
        output_preprocessing : str, optional
            [description], by default 'none'
        verbose : bool, optional
            [description], by default False
        """

        self.sample_df = sample_df['path']
        self.target_df = sample_df.drop('path',axis=1)
        
        self.num_data = len(sample_df.index)
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = dim
        self.num_channels = num_channels
        self.num_classes = num_reg_classes
        self.verbose = verbose        

        self.output_preprocessing = output_preprocessing
        self.output_scaler_inst = output_scaler
        
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
                
            if self.output_preprocessing == 'standard' or self.output_preprocessing == 'minmax' or self.output_preprocessing == 'quantile':
                transformed  = self.output_scaler_inst.fit_transform(self.target_df)
                self.target_df = pd.DataFrame(transformed, index=self.target_df.index)
            
        else:
            transformed  = self.output_scaler_inst.fit_transform(self.target_df)
            self.target_df = pd.DataFrame(transformed, index=self.target_df.index)
    

    def get_scaler_instance(self):
        return self.output_scaler_inst


    def on_epoch_end(self):
        """
        Shuffle the indices when shuffle==True. otherwise keep the index order
        """
        self.indices = np.arange(self.num_data)
        if self.shuffle:
            np.random.shuffle(self.indices)


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

        for i in range(0, self.batch_size):
            
            idx = indices[i]            
            X[i] = nib.load(self.sample_df.iloc[idx]).get_fdata().reshape((*self.dim,1))
            Y[i] = self.target_df.iloc[idx].to_numpy()

        return X, Y
            










