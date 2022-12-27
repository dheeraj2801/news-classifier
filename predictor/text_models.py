import torch
import numpy as np
from manage import TransformerCNN
embed_weight = torch.from_numpy(np.zeros((12784,300)))

MAX_LENGTH = 32
def get_models():
    trans_cnn_model = TransformerCNN(   
                            embed_weights=embed_weight,     
                            n_heads = 30,                   
                            n_feed = 1024,
                            dropout = 0.2, 
                            n_layers = 3, 
                            seq_length = MAX_LENGTH,        
                            kernal_size = 3     
                        )
    #trans_lstm_model = TransformerRNN(
                        #embed_weight = embed_weight,
                        #n_heads = 3,
                        #n_feed = 1024,
                        #dropout = 0.1, 
                        #n_layers = 3, 
                        #seq_length = MAX_LENGTH,
                        #hidden_size=128
                    #)
    #trans_lstm_model = torch.load(
                #"media/model_weights/embed_trans_lstm_21.pt")

    #trans_lstm_model.eval()

    trans_cnn_model = torch.load(
                "media/model_weights/cnn_best_cpu.pt").cpu()                #Loading the CNN Model into Transformer Model
    
    trans_cnn_model.eval()                                  #Processing the Model
    
    return trans_cnn_model #,trans_lstm_model

    