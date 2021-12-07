from typing import Tuple
import nibabel as nib
import numpy as np
import math
from tensorflow.keras.utils import Sequence


class VolumeDataGeneratorRegression(Sequence):
    def __init__(
        self, 
        sample_df, 
        batch_size=16, 
        shuffle=True, 
        dim=(160, 160, 91), 
        num_channels=1,
        num_reg_classes=1,
        verbose=False):
        """Volume data generator for regression task

        Parameters
        ----------
        sample_list : [type]
            [description]
        dir : [type]
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
        
        self.on_epoch_end()


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
            










