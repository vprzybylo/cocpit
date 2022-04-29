import pandas as pd
from cocpit import config as config
from cocpit import pic as pic
import matplotlib.pyplot as plt
import os


def main() -> None:
    """
    Reinforce labeled dataset by saving model predicted images to specified class directory
    from a specific campaign
    """
    num_images = 10  # how many images to save from each campaign
    campaigns = ["OLYMPEX"]
    desired_size = 1000

    for campaign in campaigns:
        print(f"saving images from {campaign}...")
        df = pd.read_csv(f"{config.FINAL_DIR}{campaign}.csv")
        for file, class_ in zip(
            df["filename"][:num_images], df["classification"][:num_images]
        ):
            image = pic.Image(
                f"{config.BASE_DIR}/cpi_data/campaigns/{campaign}/single_imgs_{config.TAG}/",
                file,
            )
            image.resize_stretch(desired_size)
            save_dir = f"{config.BASE_DIR}/cpi_data/campaigns/{campaign}/build_training/{class_}/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            image.save_image(save_dir)


if __name__ == "main":
    main()
