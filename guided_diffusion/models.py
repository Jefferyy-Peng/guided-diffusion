from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
from PIL import Image
import requests
from torch import nn

class Mobilevit(nn.Module):
    def __init__(self):
        super(Mobilevit, self).__init__()
        self.feature_extractor = MobileViTFeatureExtractor.from_pretrained('Matthijs/mobilevit-small')
        self.model = MobileViTForImageClassification.from_pretrained('Matthijs/mobilevit-small')

    def forward(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return self.model(**inputs)