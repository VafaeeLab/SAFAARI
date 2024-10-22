import numpy as np
import torch
import matplotlib.pyplot as plt


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(raw_data, batch_size, shuffle=True, ind=None):
    if ind is None:
        data = [raw_data['X'], raw_data['Y']]
    else:
        data = [raw_data['X'][ind], raw_data['Y'][ind]]
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield data[0][start:end], data[1][start:end]


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(6, 4))
    cmap = {0: 'k', 1: 'b', 2: 'g', 3: 'r', 4: 'y'}
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=cmap[d[i]],
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def set_random_seed(seed=0):
    # seed setting
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_hot(labels, num_classes):
    one_hot_labels = torch.zeros(labels.size(0), num_classes).to(labels.device)
    one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot_labels
