from itertools import chain
from pathlib import Path
import pickle
from random import shuffle
from re import fullmatch
from sched import scheduler

from torch.utils.data import Dataset, DataLoader, random_split
import torch

from agent import Network, XOAgentModel
from sim import TRAINING_DATA_RE


class TrainingDataset(Dataset):
    def __init__(
        self,
        revision: int | None = None,
        training_dir="training_data",
        num_to_load: int = 1024,
    ) -> None:
        self.training_dir = training_dir
        self.num_to_load = num_to_load
        self.revision = self.get_revision(revision)
        self.training_data = self.load_training_data()
        
    def get_revision(self, revision: int | None = None) -> int:
        revisions_paths = [x for x in Path(".", self.training_dir).iterdir() if x.is_dir()]
        if revision is None:
            revision = max([int(path.stem) for path in revisions_paths])
            print(f"Using latest revision: {revision}")

        return revision

    def load_training_data(self) -> dict:
        revisions_paths = [x for x in Path(".", self.training_dir).iterdir() if x.is_dir()]
        revision_path = Path(".", self.training_dir, str(self.revision))
        if revision_path not in revisions_paths:
            raise IOError(f"No saved data for revision {self.revision}")

        chunk_paths = [
            x for x in revision_path.iterdir() if x.is_file() and fullmatch(TRAINING_DATA_RE, x.name)
        ]
        chunk_idxs = [
            (int(path.stem.split("_")[-2]), int(path.stem.split("_")[-1]))
            for path in chunk_paths
        ]
        chunk_paths.reverse()
        chunk_idxs.reverse()

        to_load: list[Path] = []
        running_tot = 0
        for i, (start, stop) in enumerate(chunk_idxs):
            if running_tot < self.num_to_load:
                to_load.append(chunk_paths[i])
                num = stop - start + 1
                running_tot += num

        training_data = dict(features=list(), Y=list())
        for path in to_load:
            with open(path, "rb") as f:
                raw = pickle.load(f)
                raw = list(chain(*chain(raw)))
                features, policies, values = zip(*raw)
                Y = [
                    XOAgentModel.policy_and_value_to_model_out(policies[i], values[i])
                    for i in range(len(raw))
                ]
                training_data["features"].extend(features)
                training_data["Y"].extend(Y)

        return training_data

    def __len__(self):
        return len(self.training_data["features"])

    def __getitem__(self, index):
        return (self.training_data["features"][index], self.training_data["Y"][index])


def train_model(
    model: Network,
    revision: int | None = None,
    batch_size: int = 32,
    epochs: int = 40,
    train_test_split: float = 0.8,
    training_dir="training_data",
    num_to_load: int = 1024,
):
    all_dataset = TrainingDataset(revision, training_dir, num_to_load)
    data_revision = all_dataset.revision
    train_n = int(len(all_dataset) * train_test_split) + 1
    test_n = len(all_dataset) - train_n
    train, test = random_split(all_dataset, [train_n, test_n])
    train_dataloader, test_dataloader = DataLoader(
        train, batch_size=batch_size, shuffle=True
    ), DataLoader(test, batch_size=batch_size, shuffle=True)

    model.train()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,20], gamma=0.1)

    for epoch in range(epochs):
        last_loss = 0.0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        tot_batches = len(iter(train_dataloader))
        for batch_idx, data in enumerate(train_dataloader):
            # Every data instance is an input + label pair
            features, labels = data

            # Labels should be shape (32, 82) not (32, 1, 82)
            labels = torch.squeeze(labels, dim=1)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(features.to(torch.float32))

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            if batch_idx % 100 == 0:
                last_loss = loss.item() / batch_size  # loss per batch
                print(f"  batch {batch_idx}/{tot_batches} loss: {last_loss}")
        scheduler.step()

    torch.save(model.state_dict(), f"models/trained_model_{data_revision + 1}", )


if __name__ == "__main__":
    net = Network()

    train_model(net, epochs=40)
