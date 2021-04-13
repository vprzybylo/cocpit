import cocpit.dataloaders

def find_normalization_values():
    mean = inputs.sum()
    meansq = meansq + (inputs**2).sum()
    count += np.prod(inputs.shape)

    
train_loader, val_loader = data_loaders.create_dataloaders(data,
                                                        train_indices,
                                                        val_indices,
                                                        batch_size,
                                                        save_model,
                                                        val_loader_savename,
                                                        class_names=params['class_names'],
                                                        data_dir=params['data_dir'],
                                                        valid_size=valid_size,
                                                        num_workers=num_workers)

meansq = 0.0
mean = 0.0
count = 0
for i, ((inputs, labels, paths),index) in enumerate(dataloaders_dict[phase]):
    find_normalization_values(meansq, mean, count, inputs)