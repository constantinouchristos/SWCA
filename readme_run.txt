example to run run_swag_bidir_bert

python3 run_swag_bidir_bert.py -model_start=allenai/scibert_scivocab_uncased -out_dir=temp_experiment -epochs=3 -gms=4 -adv -avd_e=0.005 -lamb=0.5 -sw_s=0.5 -temp_c=0.7


