import torch
import os
import re

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score
from transformers import BertConfig,BertModel

from bert_lstm_model import Bert_lstm,Bert_lstm_adv
from swag_modeling import SWAG,SWAG_adversarial
from bert_modeling_custom import BertModel_c
from tqdm.auto import tqdm



def evaluate_model(model,
                   dataset,
                   my_trainer,
                   swag=False):
    
    """
    Args:
        model                          (torch.nn.module)  : model to evaluate
        my_trainer                     (Trainer class)    : dataframe of filtered input features (from constraints)
        swag                           (bool)             : wether to use swag or normal model


    Returns:
        dict:                          (dictionary)       : Dictionary of evaluated metrics
        
    """
    
    ignore_keys = None
    prediction_loss_only = None
    
    if not swag:
        model.to('cuda')
    
    # prepare data loader using trainer class
    test_dataloader = my_trainer.prepare_data(dataset)
    
    # lists to store predictions and groud trtuh values
    all_labels_true = []
    all_preds = []

    # progress bar
    prediction_bar = tqdm(total=len(test_dataloader))
    
    model.eval()
    # run evaluation
    for step, inputs in enumerate(test_dataloader):
        if my_trainer.use_adversarial:
            loss, logits, labels = my_trainer.prediction_step_bidir_adversarial(model, 
                                                                                inputs, 
                                                                                prediction_loss_only, 
                                                                                ignore_keys=ignore_keys)
        else:
            loss, logits, labels = my_trainer.prediction_step_bidir(model, 
                                                                    inputs, 
                                                                    prediction_loss_only, 
                                                                    ignore_keys=ignore_keys)
        all_preds.extend(logits.argmax(axis=1).cpu().numpy().tolist())
        all_labels_true.extend(labels.cpu().numpy().tolist())

        prediction_bar.update(1)
    
    array_true = np.array(all_labels_true)
    array_pred = np.array(all_preds)
    
    # accuracy score
    ac = accuracy_score(array_true,array_pred)
    # weighted f1 score
    f1 = f1_score(array_true,array_pred,average='weighted')
    
    return {'ac':ac,
            'f1':f1,
           }
    
    
def get_results(path_c,
                model_args,
                dataset,
                my_trainer,
                type_='normal',
                num_labels=None
               ):
    
    
    """
    Args:
        path_c                         (string)              : path ot directory of saved models
        model_args                     (model_args class)    : dataframe of filtered input features (from constraints)
        dataset                        (dataset)             : dataset for which to perform the evaluation
        type_                          (string)              : type of model to evaluate (normal or swag)

    Returns:
        result:                        (dictionary)          : Dictionary of evaluated metrics
        
    """
    
    # load model config
    config = BertConfig.from_pretrained(    
        pretrained_model_name_or_path=model_args.model_name_or_path
    )
    
    if my_trainer.use_adversarial:
        
        # BERT model
        b_model = BertModel_c.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=None
        )
        
        
        # bidirectional LSTM plus BERT adversarial
        b_model_lstm = Bert_lstm_adv(
            bert=b_model,
            bidir=True,
        )
        
    else:
        

        # load bert base model 
        b_model = BertModel.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=None
        )

        # load bert bidirectional lstm model
        b_model_lstm = Bert_lstm(
            bert=b_model,
            bidir=True,
        )
    
    
    if type_ == 'normal':

        state_dcict = torch.load(os.path.join(path_c,'pytorch_model.bin'))
        b_model_lstm.load_state_dict(state_dcict)
        result = evaluate_model(b_model_lstm,dataset,my_trainer)
        
    else:
        
        
        if my_trainer.use_adversarial:
            # swag adversarial
            swag_model = SWAG_adversarial(
                BertModel_c,
                no_cov_mat=not True,
                max_num_models=20,
                num_classes=num_labels,
                config=config,
                model_args=model_args

            )
            
        else:
                    
            swag_model = SWAG(
                BertModel,
                no_cov_mat=not True,
                max_num_models=20,
                num_classes=num_labels,
                config=config,
                model_args=model_args

            )
        
        state_dcict = torch.load(os.path.join(path_c,'swag_pytorch_model.bin'))
        swag_model.load_state_dict(state_dcict)
        swag_model.sample(0.0)
        
        result = evaluate_model(swag_model,dataset,my_trainer,swag=True)
        
    return result



def generate_evaluation(path_to_test,
                        model_args=None,
                        eval_dataset=None,
                        predict_dataset=None,
                        my_trainer=None,
                        num_labels=None
                       ):
    
    """
    Args:
        path_to_test                   (string)           : path where models were saved

    Returns:
        None:                          generates a csv file in the path_to_test directory
        
    """
    
    all_checkpoints = sorted([i for i in os.listdir(path_to_test) if 'checkpoint' in i])\

    check_point_info = []
    model_type = []
    ac_all = []
    f1_all = []
    dataset_type = []

    for dataset_t in ['eval','test']:

        if dataset_t == 'eval':
            data_to_eval = eval_dataset
        else:
            data_to_eval = predict_dataset

        for check in all_checkpoints:

            temp_check_name = re.sub('-','_',check)

            path_check = os.path.join(path_to_test,check)

            for model_t in ['normal','swag']:

                res = get_results(path_check,
                                  model_args,
                                  data_to_eval,
                                  my_trainer,
                                  type_=model_t,
                                  num_labels=num_labels
                                 )

                ac_all.append(res['ac'])
                f1_all.append(res['f1'])
                model_type.append(model_t)
                check_point_info.append(temp_check_name)
                dataset_type.append(dataset_t)

    df_results = pd.DataFrame({'checkpoint':check_point_info,
                               'model_type':model_type,
                               'ac':ac_all,
                               'f1':f1_all,
                               "data_type":dataset_type,

                              })

    df_results.to_csv(os.path.join(path_to_test,'metric_results.csv'),index=False)


