from torch import nn

class SimpleNet(nn.Module):
    def __init__(self, num_features=2048, num_concepts=21):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(num_features, num_concepts)

    def forward(self, x):
        return self.fc(x)

class ConceptNet(nn.Module):
    def __init__(self, num_features=2048, num_concepts=21):
        super(ConceptNet, self).__init__()
        intermidate_dim = 256
        self.fc = nn.Sequential(
            nn.Linear(num_features, intermidate_dim),
            nn.SiLU(),
            nn.Linear(intermidate_dim, num_concepts)
        )

    def forward(self, x):
        return self.fc(x)
