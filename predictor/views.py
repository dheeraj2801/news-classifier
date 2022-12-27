from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from .preprocess_tokenize import data_maker
from .imports import torch, p
import numpy as np
from .models import DataBase
from .text_models import get_models

trans_cnn_model=get_models()
label_decoding={11: 'ARTS & CULTURE',
    5: 'BUSINESS',
    0: 'CRIME',
    9: 'EDUCATION',
    1: 'ENTERTAINMENT',
    12: 'ENVIRONMENT',
    10: 'FAMILY & RELATIONS',
    6: 'LIVING STYLE/ HEALTH RELATED',
    2: 'POLITICS',
    8: 'RELIGION',
    4: 'SPORTS',
    7: 'TECH',
    3: 'VOICES'}
class HomePage(APIView):
    def get(self,request):
        return render(request, "web_ui.html")
class Classify(APIView):
    def post(self, request):
        try:
            ids, masks = data_maker(
                request.data["sentence"])
            #print(ids,masks,ids.shape,masks.shape)    
            label, confidence = self.predict(request.data["model_name"], ids, masks)
            print(label, confidence)
            return JsonResponse(
                {
                    "label":label,
                    "confidence":np.round(confidence*100, decimals=2)
                }
            )
        except Exception as e:
            print(e)
            return JsonResponse({
                "error":str(e)
            })
    def predict(self, model_name, inputs, masks):

        device=torch.device("cpu") #Assinging CPU memory and clock speed
        
       
        model=trans_cnn_model

        model.eval()                        #Processing the Model
        model.to(device)                    #Adjusting according to the Systems CPU (According to the Deployed Device)
        prob, target=torch.max(
                            model(
                                    inputs.to(device).type(torch.int),
                                    masks.to(device).type(torch.bool)),
                            dim=-1)                                     
        prob=p(prob)                            #Converts the result into probability 0<prob<1
        
       
        label=label_decoding[target[0].item()]      #Takes highest Probabity item
        
        # device = torch.device("cpu")        
        del model                               #To utilize the CPU
        del inputs                              #To remove the previous data
        del masks                               #To remove the previous data
        return label, prob.detach().cpu().item()        #Returning the label and the probability values

