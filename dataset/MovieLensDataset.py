import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import utils


class MovieLensDataset(Dataset):
    def __init__(self, path, library_limit=1000, request_limit=None, horizon=128, split="train", transform=None,
                 target_transform=None, mode="classification"):
        ratings = pd.read_csv(os.path.join(path + "ratings.csv"))
        library = np.array(ratings["movieId"].value_counts().reset_index())[:library_limit, 0]

        # Get movie requests based on time
        sorted_ratings = ratings.sort_values(by="timestamp")
        temp_requests = np.array(sorted_ratings["movieId"])
        # only keep requests of a movie if it is in our "library"
        requests = []
        for req in temp_requests:
            if req in library:
                requests.append(req)

        # Map movieIds to new indices
        movie_mapper = dict(zip(np.unique(requests), list(range(len(library)))))
        for i in range(len(requests)):
            requests[i] = movie_mapper[requests[i]]

        # Limit history and requests size based on parameters
        if request_limit is not None:
            requests = requests[-request_limit:]

        # Split the requests based on train or test
        if split == "train":
            requests = requests[:int(len(requests)*0.8)]
        elif split == "validation":
            requests = requests[int(len(requests)*0.8): int(len(requests)*0.9)]
        elif split == "test":
            requests = requests[int(len(requests)*0.9):]
        # Shift request by 1 into the future so that the past t-1 values should predict value at time t
        self.library_size = len(library)
        self.data = utils.convert_to_vectors(np.roll(requests, 1), library_size=self.library_size).T
        self.labels = requests[:request_limit]
        self.horizon = horizon
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Zeros are padded at the front of the request matrix if there are not enough past requests to fill the matrix
        # Matrix is of size (library size, self.horizon)
        past_idx = idx - self.horizon + 1
        requests = np.concatenate((np.zeros((self.data.shape[0], -1* past_idx if past_idx < 0 else 0)),
                                   self.data[:, np.maximum(0, past_idx):idx + 1]), axis=1)
        label = self.labels[idx]

        if self.mode == "mse":
            temp = np.zeros(self.library_size)
            temp[label] = 1
            label = temp

        if self.transform:
            requests = self.transform(requests)
        if self.target_transform:
            label = self.target_transform(label)

        return requests, label