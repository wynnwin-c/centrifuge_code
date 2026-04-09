from torch import nn


class CNN(nn.Module):
    def __init__(self, in_channel=1):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv1d(in_channel, 16, kernel_size=15, stride=1), nn.BatchNorm1d(16), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=3), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=3), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.AdaptiveMaxPool1d(4))
        self.layer5 = nn.Sequential(nn.Linear(128 * 4, 256), nn.ReLU(inplace=True), nn.Dropout())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        return x


class cnn_features(nn.Module):
    def __init__(self, in_channel=1):
        super().__init__()
        self.model_cnn = CNN(in_channel=in_channel)
        self.__in_features = 256

    def forward(self, x):
        return self.model_cnn(x)

    def output_num(self):
        return self.__in_features

    def hidden_num1(self):
        return self.__in_features // 2

    def hidden_num2(self):
        return self.__in_features // 8

    def adv_hidden_num1(self):
        return self.__in_features * 2

    def adv_hidden_num2(self):
        return self.__in_features * 2
