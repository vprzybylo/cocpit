import pandas as pd
from dataclasses import dataclass


@dataclass
class Date:
    """
    Add date to dataframe from filename

    Args:
        df (pd.DataFrame): A dataframe holding image filenames
    """

    df: pd.DataFrame

    def date_column(self) -> None:
        """
        - Create a date column from filename column
        - Note that the date format may need to change for each campaign
        """
        date_list = self.df["filename"].str.split(".").str[0].str.split("_")
        self.df["date"] = date_list.str[:-1].str.join("_")
        # print(date_list.str[:-1].str.join("_"))

    def convert_date_format(self) -> None:
        """
        Convert date column to datetime format
        """
        self.df["date"] = pd.to_datetime(
            self.df["date"], format="%Y_%m%d_%H%M%S", errors="raise"
        )

    def move_to_front(self) -> None:
        """Move the date to head of list using index, pop and insert"""
        cols = list(self.df)
        cols.insert(1, cols.pop(cols.index("date")))
        self.df = self.df.loc[:, cols]
        self.df = self.df.sort_values(by="date")
