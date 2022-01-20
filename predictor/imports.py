import re
import en_core_web_lg
import torch
import numpy as np
from torch import nn 
import torch.nn.functional as f

nlp=en_core_web_lg.load()

vocab = torch.load("media/vocabulary.pt")

MAX_LENGTH = 32

p=nn.Softmax(dim=-1)

