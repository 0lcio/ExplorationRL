import torch
import torch.nn as nn

from constants import SINGLE_MARKER_OUT, ENTROPY_OUT, LOSS_OUT, EACH_POV_MARKER_OUT, NUM_CLASSES_IG_MARKER_POV


class NetIG(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super(NetIG, self).__init__()
        self.output_size = output_size  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        if output_size == NUM_CLASSES_IG_MARKER_POV:
            self.each_pov_marker = True
        else:
            self.each_pov_marker = False

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, output_size)
        # self.sofmax = nn.Softmax(dim=1)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x.float())
        hn = hn.view(-1, self.hidden_size)

        # future prediction
        prediction = self.fc1(hn)
        prediction = self.dropout(prediction)
        prediction = self.relu(prediction)

        prediction = self.fc2(prediction)
        prediction = self.dropout(prediction)
        prediction = self.relu(prediction)

        prediction = self.fc3(prediction)
        prediction = self.relu(prediction)

        prediction = self.fc4(prediction)
        # prediction = self.relu(prediction)

        outputs = {}

        if self.each_pov_marker:
            prediction = prediction.split([SINGLE_MARKER_OUT, LOSS_OUT, ENTROPY_OUT, EACH_POV_MARKER_OUT], dim=1)

            # predizione dei marker per ogniuno dei 9 POV
            outputs['pred_marker_future_povs'] = torch.reshape(prediction[3], (9, 8))
            outputs['pred_marker_future_povs'] = self.sofmax(outputs['pred_marker_future_povs'])

        else:
            prediction = prediction.split([SINGLE_MARKER_OUT, LOSS_OUT, ENTROPY_OUT], dim=1)

        # outputs['pred_marker_last_pov'] = torch.reshape(prediction[0], (128, 8))
        outputs['pred_marker_last_pov'] = prediction[0]

        # predizione della policy1_loss per ogniuno dei 9 POV
        # outputs['pred_loss'] = torch.reshape(prediction[1], (128, 9))
        outputs['pred_loss'] = torch.exp(prediction[1])

        # predizione del'entropia per ogniuno dei 9 POV
        # outputs['pred_entropy'] = torch.reshape(prediction[2], (128, 9))
        outputs['pred_entropy'] = torch.exp(prediction[2])

        return outputs
