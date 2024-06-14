import torch.nn as nn


class MLPLtR(nn.Module):
    def __init__(self, dimensions: list):
        super().__init__()
        dimensions_model = dimensions.copy()
        dimensions_model.append(1)
        self.layers = nn.ModuleList([nn.Linear(dimensions_model[i], dimensions_model[i+1]) for i in range(len(dimensions_model) - 1)])

    def forward(self, x, return_one_to_the_last_layer=False):
        out = x
        for lay in self.layers[:-1]:
            out = lay(out)
            out = nn.functional.sigmoid(out)
        if return_one_to_the_last_layer:
            return out
        return self.layers[-1](out)

    def init_weights(self):
        def init(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)

        self.apply(init)
        return self.state_dict()


class Embed(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.l1 = nn.Linear(in_features, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)

        self.out = nn.Linear(64, 1)

    def forward(self, x):
        l1 = self.l1(x)
        o1 = nn.functional.sigmoid(l1)
        l2 = self.l2(o1)
        o2 = nn.functional.sigmoid(l2)
        l3 = self.l3(o2)
        o3 = nn.functional.sigmoid(l3)

        return o3

    def init_weights(self):
        def init(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)

        self.apply(init)
        return self.state_dict()


def get_network():
    net = Embed(136)
    net.init_weights()
    return net


