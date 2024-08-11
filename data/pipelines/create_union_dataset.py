# create_union_dataset.ipynb
# # Requirements

# %%
import os

import pickle
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# # Create union train dataset

# %%
files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
]
batches = list()
# load pickles
for filename in files:
    path_to_file = os.path.join("..", "cifar10", "cifar-10-batches-py", filename)
    with open(path_to_file, 'rb') as file:
        batch = pickle.load(file, encoding="latin1")
    batches.append(batch)
    
# Create dataset
X_train = np.concat([b["data"] for b in batches])
y_train = np.concat([b["labels"] for b in batches])

# resize X
X_train = X_train.reshape(-1, 3, 32, 32)
X_train = X_train.transpose(0, 2, 3, 1)

train = {
    "images": X_train,
    "labels": y_train
}

X_train.shape, y_train.shape

# %% [markdown]
# # Load test dataset

# %%
filename = os.path.join("..", "cifar10", "cifar-10-batches-py", "test_batch")
with open(filename, "rb") as file:
    test_batch = pickle.load(file, encoding="latin1")
X_test = test_batch["data"]
y_test = np.array(test_batch["labels"])
X_test = X_test.reshape(-1, 3, 32, 32)
X_test = X_test.transpose(0, 2, 3, 1)

test = {
    "images": X_test,
    "labels": y_test
}

X_test.shape, y_test.shape

# %% [markdown]
# # Load label encodes

# %%
filename = os.path.join("..", "cifar10", "cifar-10-batches-py", "batches.meta")
with open(filename, 'rb') as file:
    meta = pickle.load(file, encoding="latin1")
labels = meta["label_names"]
labels

# %% [markdown]
# # Save dataset

# %%
data = {
    "label_names": labels,
    "train": train,
    "test": test
}
filename = os.path.join("..", "cifar10", "data.pickle")
with open(filename, "wb") as file:
    pickle.dump(data, file)
print(f"Dataset saved as {filename}.")

# %%



