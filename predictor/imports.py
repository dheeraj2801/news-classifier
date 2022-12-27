import re
import en_core_web_lg
import torch
import numpy as np
from torch import nn 
import torch.nn.functional as f

nlp=en_core_web_lg.load() #The model (en_core_web_lg) is the largest English model of spaCy with size 788 MB.

vocab = torch.load("media/vocabulary.pt")

MAX_LENGTH = 32

p=nn.Softmax(dim=-1)

