import torch.nn as nn


class netCounterBase(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super(netCounterBase, self).__init__()
        self.output_size = output_size  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.sofmax = nn.Softmax(dim=1)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x.float())
        hn = hn.view(-1, self.hidden_size)

        # marker
        marker = self.fc1(hn)
        marker = self.dropout(marker)
        marker = self.relu(marker)

        marker = self.fc2(marker)
        marker = self.dropout(marker)
        marker = self.relu(marker)

        marker = self.relu(marker)
        marker = self.fc3(marker)

        # marker = self.sofmax(marker)

        return marker
