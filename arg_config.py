


# # custom data 2 cyclic
arguments= {"resume_from_checkpoint": False,
            "task_name":None,
             "model_name_or_path": 'allenai/scibert_scivocab_uncased',
             "output_dir": "./experiment_classification_bidir_cyclic_aversarial_exp_2/",
             "dataset_name": None,
            "dataset_config_name":None,
             "do_eval" : True,
             "do_train" : True,
             "max_seq_length": 128,
             "version_2_with_negative": True,
             "overwrite_output_dir": True,
             "num_train_epochs": 4,
             "doc_stride": 128,
             "per_device_train_batch_size": 32,
             "save_steps": 500,
             "logging_steps": 100,
             "save_total_limit": 7,
             "gradient_accumulation_steps":4,
             "fp16": True,
             "seed": 42,
            "pad_to_max_length":True,
             "train_file":'./processed/train.csv',
            "validation_file":'./processed/dev.csv',
            "test_file":'./processed/test.csv',
            "use_adversarial": True,
            "adversarial_epsilon": 0.005,
            "adv_attk_tpe": 'l2',
            "lambdaa": 0.5,
            "swag_per_start": 0.5,
            "temperature_contrastive": 0.7,
            "sample_train_p": 0,  # for reference 0.01 = 150 abstracts
            "sample_eval_p": 0,
            
            
}