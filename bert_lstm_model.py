import os
import torch.nn as nn
import torch.nn.functional as F



class Bert_lstm(nn.Module):
    
    def __init__(self,
                 bert,
                 input_dim=768, 
                 hidden_dim=768, 
                 tagset_size=5,
                 num_layers=1,
                 bidir=False
                ):
        
        super(Bert_lstm, self).__init__()
        
        self.hidden_dim_lstm = hidden_dim
        
        self.bert = bert
        self.lstm = nn.LSTM(input_dim, hidden_dim,num_layers=num_layers,bidirectional=bidir)
        
            # The linear layer that maps from hidden state space to tag space
        if bidir:
            self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
        
    def forward(self, inputs,return_h=False):
        
        inputs_bert = {i:j for i,j in inputs.items() if i != 'labels'}
        # bert ouput
        bert_pooled = self.bert(**inputs_bert)
        
        # bidirectional LSTM
        lstm_out, h = self.lstm(bert_pooled.pooler_output.view(len(bert_pooled.pooler_output), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(bert_pooled.pooler_output), -1))
        
        if return_h:
            return tag_space,h
        else:
            return tag_space
        
        
    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, 'weights.bin')
        torch.save(model_to_save.state_dict(), output_model_file)      
        


class Bert_lstm_adv(nn.Module):
    
    def __init__(self,
                 bert,
                 input_dim=768, 
                 hidden_dim=768, 
                 tagset_size=5,
                 num_layers=1,
                 bidir=False
                ):
        
        super(Bert_lstm_adv, self).__init__()
        
        self.hidden_dim_lstm = hidden_dim
        
        self.bert = bert
        self.lstm = nn.LSTM(input_dim, hidden_dim,num_layers=num_layers,bidirectional=bidir)
        
            # The linear layer that maps from hidden state space to tag space
        if bidir:
            self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
            
        self.mlp_h = nn.Linear(hidden_dim, hidden_dim)
        
        
    def forward(self, inputs,return_h=False):
        
        inputs_bert = {i:j for i,j in inputs.items() if i != 'labels'}
        
        # bert ouput and last hidden state
        bert_pooled,hidden_states = self.bert(**inputs_bert)
        
        # bidirectional LSTM
        lstm_out, h = self.lstm(bert_pooled.pooler_output.view(len(bert_pooled.pooler_output), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(bert_pooled.pooler_output), -1))
        

        # hidden states corresponding to CLS token
        first_token_tensor = hidden_states[:, 0]
        
        z_i =  F.relu(self.mlp_h(first_token_tensor))
        
        if return_h:
            return tag_space,h,z_i
        else:
            return tag_space,z_i
        
        
    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, 'weights.bin')
        torch.save(model_to_save.state_dict(), output_model_file)              