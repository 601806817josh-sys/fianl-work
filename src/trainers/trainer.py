import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.ct_dataset import CTDataset
from src.models.unet3d import UNet3D


class Trainer:
    def __init__(self, data_dir, batch_size=1, learning_rate=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = UNet3D().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.dataset = CTDataset(data_dir)

        if len(self.dataset) == 0:
            print("No CT files found in data/raw. Trainer initialization stopped.")
            self.dataloader = None
        else:
            self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0.0

        if self.dataloader is None:
            print("No training data available. Skipping training.")
            return

        for batch in self.dataloader:
            inputs = batch.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            targets = torch.zeros(
                inputs.shape[0],
                inputs.shape[2],
                inputs.shape[3],
                inputs.shape[4],
                dtype=torch.long,
                device=self.device,
            )

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            print(f"Batch loss: {batch_loss:.4f}")

        average_loss = epoch_loss / len(self.dataloader)
        return average_loss


if __name__ == "__main__":
    dataset_path = "data/raw"
    trainer = Trainer(data_dir=dataset_path, batch_size=1)
    epoch_loss = trainer.train_one_epoch()
    print(f"Epoch loss: {epoch_loss}")
