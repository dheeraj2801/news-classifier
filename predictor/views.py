from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from .preprocess_tokenize import data_maker
from .imports import torch, p
import numpy as np
from .models import DataBase
from .text_models import get_models
# Create your views here.

trans_cnn_model, trans_lstm_model = get_models()


class Classify(APIView):
    def post(self, request):
        try:
            ids, masks = data_maker(
                request.data["sentence"])
            label, confidence = self.predict(request.data["model_name"], ids, masks)

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

        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        model = trans_lstm_model if model_name =="trans_lstm" else trans_cnn_model

        model.eval()
        model.to(device)
        prob, target=torch.max(
                            model(
                                    inputs.to(device).type(torch.int),
                                    masks.to(device).type(torch.bool)),
                            dim=-1)
        prob=p(prob)
        label = "stress" if target[0]==1 else "non-stress"
        device = torch.device("cpu")
        del model
        del inputs
        del masks
        return label, prob.detach().cpu().item()

class HomePage(APIView):
    def get(self,request):
        return render(request, "web_ui.html")