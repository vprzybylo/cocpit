'''
add a column for the date from the filename
'''

import pandas as pd


def main(df):
    date_list = df['filename'].str.split(".").str[0].str.split("_")
    # create a date column from filename column
    df['date'] = date_list.str[:-2].str.join("_")
    # convert to datetime format
    df['date'] = pd.to_datetime(df['date'], format="%Y_%m%d_%H%M%S", errors='raise')

    # move the date to head of list using index, pop and insert
    cols = list(df)
    cols.insert(1, cols.pop(cols.index("date")))
    df = df.loc[:, cols]
    df = df.sort_values(by="date")
    return df
