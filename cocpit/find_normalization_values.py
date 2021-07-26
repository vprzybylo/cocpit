'''
Calculates normalization values (mean and std) on custom dataset
'''


def find_normalization_values(data):
    #     inputs.sum()
    #     meansq + (inputs ** 2).sum()
    #     np.prod(inputs.shape)

    # find mean and std for each channel, then put it in the range 0..1
    mean = np.round(data.mean(axis=(0, 1, 2)) / 255, 4)
    std = np.round(data.std(axis=(0, 1, 2)) / 255, 4)
    print(f"mean: {mean}\nstd: {std}")


train_loader, val_loader = data_loaders.create_dataloaders(
    data,
    train_indices,
    val_indices,
    batch_size,
    save_model,
    val_loader_savename,
    class_names=params["class_names"],
    data_dir=params["data_dir"],
    valid_size=valid_size,
    num_workers=num_workers,
)

meansq = 0.0
for i, ((inputs, labels, paths), index) in enumerate(dataloaders_dict[phase]):
    find_normalization_values(inputs)
