import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_num, hidden_num1, hidden_num2, output_num):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_num, output_num))

    def forward(self, x):
        return self.fc1(x)
