from pathlib import Path
import os
import pandas as pd
import math

def scanfolder(directory, sub_folder, name_wildcard, file_format) -> pd.DataFrame:
    """[summary]

    Parameters
    ----------
    directory : [type]
        [description]
    sub_folder : [type]
        [description]
    name_wildcard : [type]
        [description]d
    file_format : [type]
        [description]

    Returns
    -------
    pd.DataFrame
        [description]
    """

    paths = []
    subjects = []
    sessions = []

    for path in Path(directory).rglob('*.'+file_format):
        if (name_wildcard in str(path)) and (sub_folder in str(path)): 
            b = path.stem.split('_')
            paths.append(path)
            subjects.append(str(b[0]))
            sessions.append(str(b[1]))

    if len(paths) == 0:
        print('no files found')
        return pd.DataFrame([])
    else:
        df = pd.DataFrame(
            list(zip(subjects, sessions, paths)),
            columns =['subject', 'session', 'path'])       
        return  df

def train_test_validation_split(data_frame, train_frac=0.6, csv_write_dir=None, csv_prefix='split'):
    """[summary]

    Parameters
    ----------
    data_frame : [type]
        [description]
    train_frac : float, optional
        [description], by default 0.6
    csv_write_dir : [type], optional
        [description], by default None
    csv_prefix : str, optional
        [description], by default 'split'
    """
    df = data_frame.copy()
    sub_grouped = df.groupby('subject').nunique()
    val_test_df = sub_grouped[sub_grouped['path']==1]

    num_avail_test = len(val_test_df.index)
    num_data = len(df.index)
         
    num_train = math.floor(train_frac * num_data)
    if (num_data-num_train) > num_avail_test:
        num_train += (num_data-num_train) - num_avail_test
    num_val = math.floor(0.5 * (num_data - num_train))
    num_test = num_data - num_train - num_val

    # shuffle 
    val_test_df = val_test_df.sample(num_val+num_test)

    test_list = val_test_df[:num_test].index.to_list()
    val_list = val_test_df[num_test:num_test+num_val].index.to_list()
    
    test_df = df[df['subject'].isin(test_list)]
    valid_df = df[df['subject'].isin(val_list)]
    train_df = df[~df['subject'].isin(test_list+val_list)]

    # writing csv
    if csv_write_dir==None:
        csv_write_dir = Path.cwd()
    else:
        csv_write_dir = Path(csv_write_dir)
    
    test_path = csv_write_dir.joinpath(csv_prefix + '_test.csv')
    valid_path = csv_write_dir.joinpath(csv_prefix + '_valid.csv')
    train_path = csv_write_dir.joinpath(csv_prefix + '_train.csv')

    test_df.to_csv(test_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    train_df.to_csv(train_path, index=False) 

    return train_df, valid_df, train_df



        

        

        


if __name__ == '__main__':

    path = 'D:\Programming\school\labrotations\lab1\data\BIDS'

    path= scanfolder(path, 'anat', 'brain', 'nii.gz')
    train_test_validation_split(path)