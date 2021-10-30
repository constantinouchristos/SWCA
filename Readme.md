# SEQUENTIAL TEXT CLASSIFICATION

In this repo we tackle the task of sequential classification. The dataset considered is the pubmed 20k. The state of the art published  implementation (https://arxiv.org/pdf/1808.06161v1.pdf) can be found in this link: https://paperswithcode.com/sota/sentence-classification-on-pubmed-20k-rct.



![alt text](./images/dataset.png)




## Quick overview

The sequential classification tasks consists of 5 classes namely "METHODS", "RESULTS", "CONCLUSIONS", "BACKGROUND", "OBJECTIVE" 

The Hierarchical Neural Networks for Sequential Sentence Classification in
Medical Scientific Abstracts (Jin and Szolovits 2018)(https://arxiv.org/pdf/1808.06161v1.pdf) utilize CNNs or RNNs along with attention-based pooling to generate sentence embeddings. These are further feeded to a bidirection LSTM and final feed forward layer for the classification. My implementation replaces the CNNs/RNNs with a transformer model (Devlin,et. al 2019) (https://arxiv.org/pdf/1810.04805.pdf). Furthermore my approach incorporates techniques such as Cyclical Learning Rate by Smith (2017) (https://arxiv.org/abs/1506.01186) and Stochastic Weight Averaging (SWA) (Maddox et al. 2019)(https://arxiv.org/abs/1902.02476) to achieve a  new state of the art result.

A number of approaches were used to develop the final solution.

1) transfomer based models with feed forward Neural network (Scibert FNN) (treating sentences independently)

2) Scibert with feed forward Neural network plus positional features (Scibert FNN + pos feats )

3) Scibert with Bidirectional LSTM and feed forward Neural network plus positional features (Scibert+bidir+FNN + pos feats)

4) Scibert with Bidirectional LSTM and feed forward Neural network plus positional features plus SWA plus cyclic leanring rate (Scibert+bidir+FNN + pos feats + SWA + Cyclic)

**It must be noted that in this repo only the last approach has been implemented which is the best out of the 4**



## Requirements

To run the scripts you need a conda environment, python 3+, Pytoch and transformers from huggingface.



## Data

The data used is this study are found in the raw_data directory in compressed format. To uncompress and process data to be used for modelling use:

```
bash create_data.sh
```

The shell script calls ```run_processing.py``` and should result in the creation of a directory named **"processed"** which should consist the uncompressed raw data and there processed version in csv format




## Addtional technical details

Per the original Transformer-XL, we also implement an adaptive softmax layer (Grave et. al. 2017, https://arxiv.org/abs/1609.04309) to deal with a potentially large number of outputs in the final dense layer. This implemenation is inspired by the TF 1.0 example at https://github.com/yangsaiyong/tf-adaptive-softmax-lstm-lm.
To use the adaptive softmax, set the ```--cutoffs=``` flag in train.py. The cutoffs are the max values of each bin, and should NOT include the vocab size (i.e. the max cutoff of the final bin). If no cutoffs are specified, the model defaults to normal softmax.

For completeness, we have also provided a script ```optimal_cuts.py``` that determines the optimal cutoffs given a return space separated file of unigram probabilities (based on the assumptions of Grave et. al. regarding GPU computation complexity -- see the paper for details). 
The algorithm uses dynamic programming, but is quite slow at O(KN^2), for K cutoffs and N vocab words. In principle it's a one time cost to determine the cutoffs, but we are impatient and recommend to just play around with the cutoffs instead. See the script for flag details

## Training and Benchmarks



### transfomer based models with feed forward Neural network (Scibert FNN) (treating sentences independently)

For the initial experimentation (not included in this directory) the performance of different transfomrer models was explored namely BERT,GPT2 and SCi-BERT (BERT pretrained on published scientific documnets).

![alt text](./images/transformer_perfomance.png)


As we can see with Scibert as a back bone we achieve the highest accuracy score. This can be explained from the fact that the data used for SCi-BERT pretraining are in a similar domain (pusblished scientific papers) as the data avialable for this task

### Scibert with feed forward Neural network plus positional features (Scibert FNN + pos feats )

At this point as we are treating sentences independetly we dont capture the sequential nature of the labels withing an abstract. For example a simple data analysis will show that classes such as "BACKGROUND", "OBJECTIVE" are mostly found a the start of abstracts while classed such as "RESULTS", "CONCLUSIONS" are found towards the end.

![alt text](./images/positional_features.png)

As seen in the figure the number of sentences in an abstract can range from as low as 4 to as high as almost 30. Thus depending on the size of each abstract we can get a normalised score from 0 to 1 that represets the position of each sentence in any abstract irresspective of the length where as the position of sentence i in an abstract with totall number of sentences x is i/x. Using this normalised value we can introduced positional features for each sentence based on its normalised score. Repeating the training with Scibert and the added positional features an improvement in performance was observed

![alt text](./images/positional_feats_performance.png)


Despite the improvement in performance we still dont capture the sequential information from surrouding sentences as we still model sentences independently.

### Scibert with Bidirectional LSTM and feed forward Neural network plus positional features (Scibert+bidir+FNN + pos feats)

As a result for the next experiment we incorparate biderectional LSTMs before the final feed forward netework to model the sequential information from surrounding sentences.

![alt text](./images/bidir_performance.png)

As it can be seen this method replicated the state of the art result achieved by (Jin and Szolovits 2018)(https://arxiv.org/pdf/1808.06161v1.pdf)

But can we do better?

### Scibert with Bidirectional LSTM and feed forward Neural network plus positional features plus SWA plus cyclic leanring rate (Scibert+bidir+FNN + pos feats + SWA + Cyclic)

In order to push the performance of our method even higher we incorpate concepts such Cyclical Learning Rate by Smith (2017) (https://arxiv.org/abs/1506.01186) and Stochastic Weight Averaging (SWA) (Maddox et al. 2019)(https://arxiv.org/abs/1902.02476).


results:
![alt text](./images/bidir_performance_swag.png)



## Summary of results


![alt text](./images/summary.png)



## Training and Evaluation

To train modify any derired arguments in the ```arg_config.py```, file and then run :
```
bash run_train_evaluation.sh
```
Upon running this strict the model training should be performed with the saved models and evaluation results saved in the directory specified by the "output_dir" argument of ```arg_config.py```. the script will run ```run_swag_bidir_bert.py```



## Thanks for reading this far!

Enjoy! And thank you to the wonderful researchers that inspired this project.

If you have any comments or questions email me directly at christos1361993@gmail.com