import os

import cocpit

# flip images for training
open_dir = "cpi_data/training_datasets/rimed_agg/"
save_dir = "cpi_data/training_datasets/rimed_agg_flip/"
for file in os.listdir(open_dir):
    image = cocpit.pic.Image(open_dir, file)
    image.save_image(save_dir, flip=True)
