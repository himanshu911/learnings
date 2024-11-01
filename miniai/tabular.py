import torch
from torch import nn
from torch.utils.data import DataLoader

from miniai import (
    Dataset,
    DataLoaders,
    Learner,
    TrainCB,
    MetricsCB,
    DeviceCB,
    ProgressCB,
)

from miniai import to_device

from utils import load_sklearn_dataset


class RegressionNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128, dtype=torch.float64)
        self.fc2 = nn.Linear(128, 64, dtype=torch.float64)
        self.fc3 = nn.Linear(64, output_size, dtype=torch.float64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)


def prepare_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    # Create Dataset instances
    train_ds = Dataset(X_train, y_train)
    valid_ds = Dataset(X_val, y_val)
    test_ds = Dataset(X_test, y_test)

    # Prepare DataLoaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size * 2)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

    return train_dl, val_dl, test_dl


def train_model(model, train_dl, val_dl, lr, n_epochs):
    # Prepare the learner
    loss_func = nn.MSELoss()
    opt_func = torch.optim.Adam
    cbs = [
        TrainCB(),
        DeviceCB(),
        MetricsCB(),
        # ProgressCB(),
    ]
    learner = Learner(
        model, DataLoaders(train_dl, val_dl), loss_func, lr, opt_func=opt_func, cbs=cbs
    )

    # Train the model
    learner.fit(n_epochs=n_epochs, train=True, valid=True)
    return learner


def evaluate_model(learner, test_dl):
    learner.model.eval()  # Set the model to evaluation mode
    total_loss = 0
    loss_func = nn.MSELoss()
    device = next(learner.model.parameters()).device
    with torch.no_grad():  # No need to track gradients for evaluation
        for xb, yb in test_dl:
            xb, yb = to_device(xb, device), to_device(yb, device)
            preds = learner.model(xb)
            loss = loss_func(preds, yb)
            total_loss += loss.item() * len(xb)

    mean_loss = total_loss / len(test_dl.dataset)
    print(f"Mean Squared Error on Test Set: {mean_loss}")


# Main execution block
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_sklearn_dataset(
        "california_housing"
    )
    batch_size = 64
    n_epochs = 10
    lr = 0.01

    train_dl, val_dl, test_dl = prepare_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size
    )

    # Initialize the model
    model = RegressionNN(input_size=X_train.shape[1], output_size=1)

    # Train the model
    learner = train_model(model, train_dl, val_dl, lr, n_epochs)

    # Evaluate the trained model on the test set
    evaluate_model(learner, test_dl)
