import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.neighbors import RadiusNeighborsRegressor


class StateValueResidual(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(StateValueResidual, self).__init__()
        # Store args
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Hidden layer size
        self.hidden = 64

        # Layers
        self.fc1 = nn.Linear(self.in_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.hidden)
        self.v_out = nn.Linear(self.hidden, self.out_dim)
        # Initialize the last layer weights to be zero
        self.v_out.weight.data = torch.zeros(self.out_dim, self.hidden)
        self.v_out.bias.data = torch.zeros(self.out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.v_out(x)

        return value


def get_state_value_residual(features, state_value_residual):
    features_tensor = torch.from_numpy(features).view(1, -1)
    with torch.no_grad():
        residual_tensor = state_value_residual(features_tensor)
        residual = residual_tensor.detach().cpu().numpy().squeeze()

    return residual


class StateActionValueResidual(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(StateActionValueResidual, self).__init__()
        # Store args
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Hidden layer size
        self.hidden = 64

        # Layers
        self.fc1 = nn.Linear(self.in_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.hidden)
        self.q_out = nn.Linear(self.hidden, self.out_dim)
        # Initialize last layer weights to be zero
        self.q_out.weight.data = torch.zeros(self.out_dim, self.hidden)
        self.q_out.bias.data = torch.zeros(self.out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.q_out(x)

        return q_values


def get_state_action_value_residual(features, ac_idx, state_action_value_residual):
    features_tensor = torch.from_numpy(features).view(1, -1)
    with torch.no_grad():
        residual_tensor = state_action_value_residual(features_tensor)
        residual = residual_tensor.detach().cpu().numpy().squeeze()

    return residual[ac_idx]


class DynamicsResidual(nn.Module):
    def __init__(self, in_dim, num_actions, out_dim):
        super(DynamicsResidual, self).__init__()
        # Store args
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_actions = num_actions

        self.in_dim = self.in_dim + self.num_actions

        self.hidden = 64

        # Layers
        self.fc1 = nn.Linear(self.in_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.dyn_out = nn.Linear(self.hidden, self.out_dim)
        # Initialize last layer weights to be zero
        self.dyn_out.weight.data = torch.zeros(self.out_dim, self.hidden)
        self.dyn_out.bias.data = torch.zeros(self.out_dim)

    def forward(self, s, a):
        one_hot_encoding = torch.zeros(
            a.shape[0], self.num_actions)
        for row in range(a.shape[0]):
            one_hot_encoding[row, a[row]] = 1
        x = torch.cat([s, one_hot_encoding], dim=1)
        dyn_x = F.relu(self.fc1(x))
        dyn_x = F.relu(self.fc2(dyn_x))
        dyn_values = self.dyn_out(dyn_x)
        return dyn_values


def get_dynamics_residual(representation, ac_idx, dynamics_residual):
    representation_tensor = torch.from_numpy(representation).view(1, -1)
    ac_idx_expanded = np.array([ac_idx], dtype=np.int32)
    with torch.no_grad():
        dynamics_residual_tensor = dynamics_residual(representation_tensor,
                                                     ac_idx_expanded)
        dynamics_residual = dynamics_residual_tensor.detach().cpu().numpy().squeeze()

    return dynamics_residual


class KNNDynamicsResidual:
    def __init__(self, in_dim, radius, out_dim):
        # Save args
        self.in_dim = in_dim
        self.radius = radius
        self.out_dim = out_dim
        # Create the KNN model
        self.knn_model = RadiusNeighborsRegressor(radius=radius,
                                                  weights='uniform',
                                                  metric='manhattan')
        # Flag
        self.is_fit = False

    def fit(self, X, y):
        self.knn_model.fit(X, y)
        self.is_fit = True
        return self.loss(X, y)

    def predict(self, X):
        '''
        X should be the data matrix N x d, where each row is a 4D vector
        consisting of object pos and gripper pos
        '''
        ypred = np.zeros((X.shape[0], self.out_dim))
        if not self.is_fit:
            # KNN model is not fit
            return ypred
        # Get neighbors of X
        neighbors = self.knn_model.radius_neighbors(X)
        # Check if any of the X doesn't have any neighbors by getting nonzero mask
        neighbor_mask = [x.shape[0] != 0 for x in neighbors[1]]
        # If none of X has any neighbors
        if X[neighbor_mask].shape[0] == 0:
            return ypred
        # Else, for the X that have neighbors use the KNN prediction
        ypred[neighbor_mask] = self.knn_model.predict(X[neighbor_mask])
        return ypred

    def get_num_neighbors(self, X):
        if not self.is_fit:
            return np.zeros(X.shape[0])
        neighbors = self.knn_model.radius_neighbors(X)
        num_neighbors = np.array([x.shape[0] for x in neighbors[1]])
        return num_neighbors

    def loss(self, X, y):
        ypred = self.predict(X)
        # Loss is just the mean distance between predictions and true targets
        loss = np.linalg.norm(ypred - y, axis=1).mean()
        return loss


def get_knn_dynamics_residual(representation, knn_dynamics_residual):
    representation_input = representation.reshape(1, -1)
    dynamics_residual = knn_dynamics_residual.predict(
        representation_input).squeeze()
    return dynamics_residual
