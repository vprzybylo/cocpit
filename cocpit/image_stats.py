"""
find the #/% of cutoff particles after removing blurry, fragmented, and spherical drops
"""
import pandas as pd


def main():
    campaigns = [
        "AIRS_II",
        "ARM",
        "ATTREX",
        "CRYSTAL_FACE_NASA",
        "CRYSTAL_FACE_UND",
        "ICE_L",
        "MACPEX",
        "MC3E",
        "MIDCIX",
        "MPACE",
        "OLYMPEX",
        "POSIDON",
    ]

    cutoff = 10.0
    for campaign in campaigns:
        df = pd.read_csv("/data/data/final_databases/vgg19/" + campaign + ".csv")
        len_df_less_cutoff = len(df[df["cutoff"] >= cutoff])
        print(
            campaign,
            " #/% cuttoff >= "
            + str(cutoff)
            + ": %.2d %.2f"
            % (len_df_less_cutoff, (len_df_less_cutoff / len(df)) * 100),
        )


if __name__ == '__main__':
    main()
