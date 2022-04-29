from cocpit import config as config
import pandas as pd


def main(cutoff: float = 10.0) -> None:
    """
    Print the #/% of cutoff particles after removing blurry, fragmented, and spherical drops

    Args:
        cutoff (float): Percent of particle that can intersect the border
    """
    for campaign in config.CAMPAIGNS:
        df = pd.read_csv("/data/data/final_databases/vgg19/" + campaign + ".csv")
        len_df_less_cutoff = len(df[df["cutoff"] >= cutoff])
        print(
            campaign,
            " #/% cuttoff >= "
            + str(cutoff)
            + ": %.2d %.2f"
            % (len_df_less_cutoff, (len_df_less_cutoff / len(df)) * 100),
        )


if __name__ == "__main__":
    main()
