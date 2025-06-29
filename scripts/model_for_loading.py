import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l

class SameDifferentCNN_from_checkpoint(nn.Module):
    """
    This class reconstructs the specific architecture required to load the
    `maml_pbweights_conv6/model_seed_4_pretesting.pt` checkpoint.
    
    The architecture was deduced from the RuntimeError encountered when
    trying to load the state_dict with the wrong model.
    """
    def __init__(self):
        super(SameDifferentCNN_from_checkpoint, self).__init__()
        
        # Layer dimensions are taken directly from the error message's size mismatches
        self.conv1 = nn.Conv2d(3, 6, kernel_size=2, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6, track_running_stats=False)
        
        self.conv2 = nn.Conv2d(6, 12, kernel_size=2, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(12, track_running_stats=False)

        # The original model was missing layers 3-6, so we will omit them.
        
        self.pool = nn.MaxPool2d(2) # A default pooling layer
        
        # This is a placeholder to calculate the flattened size. 
        # The actual forward pass will need to be determined.
        self._to_linear = 13068 # From the error: fc_layers.0.weight has shape [1024, 13068]

        # FC layers with dimensions from the error
        self.fc_layers = nn.ModuleList([
            nn.Linear(self._to_linear, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024)
        ])
        
        # Layer norms matching the FC layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(1024),
            nn.LayerNorm(1024),
            nn.LayerNorm(1024)
        ])
        
        self.classifier = nn.Linear(1024, 2)
        
        # The learnable LRs are expected by the state_dict
        self.lr_conv = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(2) # Only 2 conv layers
        ])
        self.lr_fc = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(3)
        ])
        self.lr_classifier = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # NOTE: This forward pass is a placeholder based on common CNN designs.
        # It may not perfectly match the original model's logic, but it allows
        # us to load the weights and inspect the layers.
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten the features for the fully connected layers.
        # Use .reshape() instead of .view() to handle non-contiguous tensors.
        x = x.reshape(x.size(0), -1)
        
        # The size might not match _to_linear perfectly due to pooling. If so, a
        # size mismatch error will occur on the first Linear layer below.
        if x.shape[1] != self._to_linear:
            raise RuntimeError(f"FATAL: Flattened size ({x.shape[1]}) does not match expected linear layer input size ({self._to_linear}). The reconstructed forward pass is incorrect.")

        for fc, ln in zip(self.fc_layers, self.layer_norms):
            x = F.relu(ln(fc(x)))
            
        return self.classifier(x) 