import torch.nn as nn

class SpeculatorLMHead(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)