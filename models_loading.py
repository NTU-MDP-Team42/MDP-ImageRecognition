import torch

week8 = torch.hub.load(
    './yolov5', 'custom', path='pytorch-models/Week_8.pt', source='local'
)

week9 = torch.hub.load(
    './yolov5', 'custom', path='pytorch-models/Week_9.pt', source='local'
)