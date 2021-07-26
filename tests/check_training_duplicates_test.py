import os

f = open("training_duplicates.txt", "a")
open_dir = "cpi_data/training_datasets/hand_labeled_resized_multcampaigns_clean/"

for folder in os.listdir(open_dir):
    if folder == "aggs":
        print(folder)
        count_2000_ARM = 0
        count_NASA = 0
        count_AIRS = 0
        count_ICEL = 0
        count_Midcix = 0
        count_UND = 0
        count_OLYMPEX = 0

        for file in os.listdir(open_dir + folder):
            if file in os.listdir("cpi_data/campaigns/2000_ARM/single_imgs"):
                print("file in 2000_ARM")
                count_2000_ARM += 1
            if file in os.listdir(
                "cpi_data/campaigns/2002_CRYSTAL-FACE-NASA/single_imgs"
            ):
                count_NASA += 1
            if file in os.listdir(
                "cpi_data/campaigns/2002_CRYSTAL-FACE-UND/single_imgs"
            ):
                print("file in 2002_CRYSTAL-FACE-UND")
                count_UND += 1
            if file in os.listdir("cpi_data/campaigns/2003_AIRS_II/single_imgs"):
                print("file in 2003_AIRS_II")
                count_AIRS += 1
            if file in os.listdir("cpi_data/campaigns/2004_Midcix/single_imgs"):
                print("file in 2004_Midcix")
                count_Midcix += 1
            if file in os.listdir("cpi_data/campaigns/2007_ICE_L/single_imgs"):
                print("file in 2007_ICE_L")
                count_ICEL += 1
            if file in os.listdir("cpi_data/campaigns/OLYMPEX/single_imgs"):
                print("file in OLYMPEX")
                count_OLYMPEX += 1

        print(
            count_2000_ARM,
            count_NASA,
            count_UND,
            count_AIRS,
            count_Midcix,
            count_ICEL,
            count_OLYMPEX,
        )
        total_count = (
            count_2000_ARM
            + count_NASA
            + count_UND
            + count_AIRS
            + count_Midcix
            + count_ICEL
            + count_OLYMPEX
        )
        print("total", total_count)

        f.write("ARM, NASA, UND, AIRS, Midcix, ICEL, Olympex \n")
        f.write(
            "%s,%s,%s,%s,%s,%s,%s\n"
            % (
                count_2000_ARM,
                count_NASA,
                count_UND,
                count_AIRS,
                count_Midcix,
                count_ICEL,
                count_OLYMPEX,
            )
        )
        f.write(total_count)

f.close()
