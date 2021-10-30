from processing_utils import *
import os
import gzip
import shutil



if __name__ == '__main__':
    
    
    path_data_read = './raw_data/'
    path_data_processed = './processed/'
    
    
    # if path_data_processed doesn't exit create it
    if not os.path.exists(path_data_processed):

        os.makedirs(path_data_processed)

    # uncompress data and save them into path_data_processed
    uncompress_data(path_data_read,path_data_processed)
    
    
    # paths uncopmressed data
    paths_uncompressed_data = [os.path.join(path_data_processed,i) for i in os.listdir(path_data_processed) if '.txt' in i]

    # paths in dictionary
    data_dict_paths = {os.path.splitext(os.path.basename(i))[0]:i for i in paths_uncompressed_data}


    # convert txt to dictionary
    diction_data = {i:txt_to_dict(data_dict_paths[i]) for i in data_dict_paths}


    # convert dictionary to pandas with all sequences and there labels
    dfs_labs = {i:get_dataframe_passage_labels(diction_data[i]) for i in diction_data}

    # convert dictionary to pandas with respect to each abstract
    dfs_abstr = {i:get_dataframe_passage(diction_data[i]) for i in diction_data}


    # process data to add positional features
    dfs_labs_positional = {i:make_pos_feats(dfs_labs[i]) for i in dfs_labs}

    # merging seq and positional feature columns
    merged_dfs_labs_positional = {i:combine_positialn_feat_with_seq(dfs_labs_positional[i]) for i in dfs_labs_positional}

    # saving processed data
    [merged_dfs_labs_positional[i].to_csv(os.path.join(path_data_processed,i+'.csv'),index=False) for i in merged_dfs_labs_positional
    ]


    