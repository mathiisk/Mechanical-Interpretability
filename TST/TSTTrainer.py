import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim

from torch.utils.data import DataLoader, TensorDataset
from aeon.datasets import load_classification

X_train, y_train = load_classification("JapaneseVowels", split="train")
X_test, y_test = load_classification("JapaneseVowels", split="test")

print("Unique y_train values:", np.unique(y_train))
print("Unique y_test values:", np.unique(y_test))
print("Original X_train shape:", X_train.shape)

# The dataset is returned as (samples, 1, seq_len). We swap axes to obtain (samples, seq_len, 1)
X_train_np = X_train.astype(np.float32)
X_train_np = np.swapaxes(X_train_np, 1, 2)
X_test_np = X_test.astype(np.float32)
X_test_np = np.swapaxes(X_test_np, 1, 2)

print("Reshaped X_train shape:", X_train_np.shape)

# Process targets: Convert labels to integers and remap so that they start at 0.
y_train = np.array(y_train).astype(np.int64)
y_test = np.array(y_test).astype(np.int64)
min_val = y_train.min()
if min_val != 0:
    y_train = y_train - min_val
    y_test = y_test - min_val

# Convert numpy arrays to torch tensors.
X_train = torch.tensor(X_train_np)
X_test = torch.tensor(X_test_np)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

batch_size = 4
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len, d_model, nhead, num_layers,
                 dim_feedforward, dropout):
        super(TimeSeriesTransformer, self).__init__()
        self.seq_len = seq_len

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=d_model // 4, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(d_model // 4)
        self.conv2 = nn.Conv1d(in_channels=d_model // 4, out_channels=d_model // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model // 2)
        self.conv3 = nn.Conv1d(in_channels=d_model // 2, out_channels=d_model, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(d_model)

        # Learnable positional encoding: shape (1, seq_len, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Transformer encoder with batch_first=True for easier tensor handling
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global pooling across time: The transformer output shape: (B, seq_len, d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)


    def forward(self, x):
        # x: (B, seq_len, input_dim)
        # Transpose for convolution: (B, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.transpose(1, 2)  # (B, seq_len, d_model)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)
        logits = self.classifier(pooled)
        return logits


    def train_epoch(self, dataloader, optimizer, loss_fn, device):
        self.train()
        running_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = self(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss


    def evaluate(self, dataloader, device):
        self.eval()
        correct = 0
        total = 0
        predictions = []
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = self(X_batch)
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy().tolist())
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        accuracy = correct / total
        return accuracy, predictions


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = X_train.shape[1]   # (e.g. 426)
    input_dim = X_train.shape[2]   # (1 for univariate)
    num_classes = len(np.unique(y_train))  # Number of classes in JapaneseVowels

    model = TimeSeriesTransformer(input_dim, num_classes, seq_len, d_model=128, nhead=8,
                                  num_layers=3, dim_feedforward=256, dropout=0.1).to(device)

    model_path = "time_series_transformer_fancy.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded saved model from", model_path)
    else:
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        optimizer = optim.RAdam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        num_epochs = 100  # Increase epochs if needed
        for epoch in range(1, num_epochs + 1):
            train_loss = model.train_epoch(train_loader, optimizer, loss_fn, device)
            # scheduler.step()  # Step the scheduler after each epoch
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}, train loss: {train_loss:.4f}, current lr: {current_lr:.6f}")
        torch.save(model.state_dict(), model_path)
        print("Saved model to", model_path)

    test_accuracy, test_preds = model.evaluate(test_loader, device)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

