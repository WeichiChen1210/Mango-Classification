import torch
import torch.nn as nn

__all__ = ['EnsembleFC', 'EnsembleFV']

class EnsembleFC(nn.Module):
    def __init__(self, model_list, num_classes=3):
        super(EnsembleFC, self).__init__()
        self.model_list = model_list
        self.fc = nn.Linear(num_classes * len(self.model_list), num_classes)
    
    def forward(self, x):
        for model in self.model_list:
            model.eval()
        
        ensemble = []
        with torch.no_grad():            
            for idx, model in enumerate(self.model_list):
                outputs, _ = model(x)
                ensemble.append(sum(outputs) / len(outputs))
    
        out = torch.cat(ensemble, dim=1)
        out = self.fc(out)
        return out
    
class EnsembleFV(nn.Module):
    def __init__(self, model_list, num_classes=3):
        super(EnsembleFV, self).__init__()
        self.model_list = model_list
        in_num = 0
        for model in model_list:
            in_num += model.backbone.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(in_num, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes))
    
    def forward(self, x):
        for model in self.model_list:
            model.eval()
        
        ensemble = []
        with torch.no_grad():
            for idx, model in enumerate(self.model_list):
                features = model(x, False)
                ensemble.append(sum(features)/len(features))
    
        out = torch.cat(ensemble, dim=1)
        out = self.fc(out)
        return out