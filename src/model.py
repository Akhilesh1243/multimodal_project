import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision.models import resnet50, ResNet50_Weights

class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes=2, text_model_name="distilbert-base-uncased",
                 text_hidden=768, img_hidden=2048, fusion_hidden=512, dropout=0.2):
        super().__init__()
        
        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)

       
        weights = ResNet50_Weights.DEFAULT
        self.image_encoder = resnet50(weights=weights)
        self.image_encoder.fc = nn.Identity()  

        self.fusion = nn.Sequential(
            nn.Linear(text_hidden + img_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes)
        )

    def forward(self, input_ids, attention_mask, images, token_type_ids=None):
        t = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = t.last_hidden_state[:, 0, :]              
        img_feat = self.image_encoder(images)                 
        x = torch.cat([text_feat, img_feat], dim=1)           
        logits = self.fusion(x)                                
        return logits
