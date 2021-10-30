import os 
import pandas as pd
import zipfile
import re
import numpy as np
import gzip
import shutil


def uncompress_data(path_c,
                    path_extract):
    
    """
    Args:
        path_c                  (str)   : path where compresed data are stored
        path_extract            (str)   : path where to save uncompressed data

    Returns:
        None                            : uncompresses files to desired path
        
    """
    
    raw_data_paths = [os.path.join(path_c,i) for i in os.listdir(path_c) if '.gz' in i]
    
    for i in raw_data_paths:

        base_name = os.path.basename(i)
        unzip_name,_ = os.path.splitext(base_name)
        new_file_name = os.path.join(path_extract,unzip_name)

        with gzip.open(i, 'rb') as f_in:
            with open(new_file_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def txt_to_dict(train_file):

    """
    Args:
        train_file                  (str)           : path to text file

    Returns:
        examples:                   (dict)          : dictionary of data
        
    """

    examples = {}

    with open(train_file, "r") as f:

        for line in f.readlines():

            if line[0] == '#':
                
                id_num = re.findall('\d+',line.strip())[0]
                examples[id_num] = {}
                temp_seqs = []
                temp_labels = []
            else:

                if line == '\n':
                    examples[id_num]['seqs'] = temp_seqs
                    examples[id_num]['labels'] = temp_labels

                else:

                    class_label,seq = line.strip().split('\t')
                    temp_seqs.append(seq)
                    temp_labels.append(class_label)
                    
    return examples
    
def get_dataframe_passage(examples):
    
    """
    Args:
        examples                    (dict)          : dictionary data for each abstract

    Returns:
        df:                         (pd.DataFrame)  : dataframe consisting of data
        
    """
    
    examples_keys = list(examples.keys())
    
    paragraphs = []
    label_sequences = []
    
    for i in examples_keys:
        
        all_seq_i = examples[i]['seqs']
        all_seq_labels = examples[i]['labels']
        
        paragraphs.append(' '.join(all_seq_i))
        label_sequences.append(all_seq_labels)
        
    df = pd.DataFrame({'un_id':examples_keys,
                       'parag':paragraphs,
                       'lab_s':label_sequences,
                      })
    
    return df


def get_dataframe_passage_labels(examples):
    """
    Args:
        examples                    (dict)          : dictionary data for each abstract

    Returns:
        df:                         (pd.DataFrame)  : dataframe consisting of data
        
    """
    
    examples_keys = list(examples.keys())
    
    idss = []
    seqs = []
    labs = []
    
    for i in examples_keys:
        
    
        all_seq_i = examples[i]['seqs']
        all_labels_i = examples[i]['labels']
        all_ids = [i for j in all_labels_i]
        
        idss.extend(all_ids)
        seqs.extend(all_seq_i)
        labs.extend(all_labels_i)
        
        
    df = pd.DataFrame({'un_id':idss,
                       'seqs':seqs,
                       'labs':labs,
                      })
    
    return df


def get_positional_feats(df,
                         uni_par_id
                        ):
    
    """
    Args:
        df                          (pd.DataFrame)  : dataframe consisting of data
        uni_par_id                  (integer)       : unique abstract id

    Returns:
        original_pos:               (list)          : positional features for a given abstract
        
    """
    
    temp_df = df[df.un_id == uni_par_id].copy()
    num_sentences = len(temp_df)
    num_sentences -= 2
    
    if num_sentences == 2:
        positional_info = ['middle','end']
        to_devide = len(positional_info)
        
    elif num_sentences == 1:
        positional_info = ['middle']
        to_devide = len(positional_info)
        
    elif num_sentences == 3:
        positional_info = ['middle','middle to end','end']
        to_devide = len(positional_info)
        
    elif num_sentences == 4:
        positional_info = ['start to middle', 'middle','middle to end','end']
        to_devide = len(positional_info)
        
    else:
        positional_info = ['start','start to middle', 'middle','middle to end','end']
        to_devide = len(positional_info)

    bins_size = num_sentences // to_devide
    remander = num_sentences % to_devide
    
    priority = ['middle','start to middle','middle to end','end','start']
    priority_filtered = priority[:remander]
    
    original_pos = np.array([[i for j in range(bins_size)] for i in positional_info]).flatten()

    
    for k in priority_filtered:

        pos_index_max = np.where(original_pos == k)[0][-1]
        initial_arr = original_pos[:pos_index_max]
        end_arr = original_pos[pos_index_max:]
        original_pos = np.concatenate([initial_arr,[k],end_arr],axis=0)

    
    original_pos = ['first position']+ list(original_pos) + ['last position']

    return original_pos

def make_pos_feats(df):
    
    """
    Args:
        df                          (pd.DataFrame)  : dataframe consisting of data

    Returns:
        df                          (pd.DataFrame)  : dataframe consisting of data
        
    """
    pos_feats = []

    for p,i in enumerate(df.un_id.unique()):

        original_pos_f2 = get_positional_feats(df,i)
        pos_feats.extend(original_pos_f2)
        
    df['pos_feat'] = pos_feats
    
    return df


def combine_positialn_feat_with_seq(df):
    
    
    """
    Args:
        df                  (pd.DataFrame)  : dataframe of data

    Returns:
        df:                 (pd.DataFrame)  : dataframe of data with merged features
        
    """
    
    df['seqs'] = 'at ' + df.pos_feat + ' ' +  df.seqs
    df.drop(['pos_feat'],axis=1,inplace=True)
    
    return df
    
    