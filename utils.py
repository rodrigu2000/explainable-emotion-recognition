import torch

def compute_stats_ds(dataset):
    mean, std = torch.zeros(3), torch.zeros(3)
    for image, labe in dataset:
        mean += image.mean(dim=(1, 2))
        std += image.std(dim=(1, 2))

    mean /= len(dataset)
    std /= len(dataset)

    print(f"mean: {mean}")
    print(f"std: {std}")
