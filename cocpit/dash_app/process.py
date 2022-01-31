'''CPI campaign data processing functions for dataframes'''
import pandas as pd
import numpy as np
import globals


def read_campaign(campaign):
    '''read particle property df and environmental property df_env
    merge based on filename and date'''

    df_env = pd.read_csv(
        f"../../final_databases/vgg16/v1.4.0/environment/{campaign}.csv",
        names=globals.col_names_env,
        header=0,
    )
    df = pd.read_csv(
        f"../../final_databases/vgg16/v1.4.0/{campaign}.csv",
        names=globals.col_names,
        header=0,
    )
    df = pd.merge(df, df_env, on=['filename', 'date'])
    return df


def remove_bad_props(df):
    '''remove bad data for particle geometric properties (e.g., no particle area)'''
    df = df[df["Area Ratio"] != -999.0]
    df = df[df["Complexity"] != 0.0]
    df = df[df["Complexity"] != -0.0]

    df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    df.dropna(inplace=True)
    return df


def remove_bad_env(df):
    '''remove missing or bad environmental data'''
    df = df[
        (df['Latitude'] != -999.99)
        & (df['Latitude'] != 0)
        & (df['Longitude'] != -999.99)
        & (df['Longitude'] != 0)
        & (df['Altitude'] != -999.99)
        & (df['Temperature'] != -999.99)
        & (df['Pressure'] != 0)
        & (df['Pressure'] != -999.99)
        & (df['Temperature'].notna())
        & (df['Ice Water Content'] != -999.99)
        & (df['Ice Water Content'].notna())
        & (df['Ice Water Content'] > 1e-5)
    ]
    return df


def check_date_range(df, start_date, end_date):
    df['date'] = df['date'].str.split(' ').str[0]
    df = df[df['date'].between(start_date, end_date)]
    return df


def check_temp_range(df, min_temp, max_temp):
    '''find temperature within a user range from input'''
    df = df[df['Temperature'] >= int(min_temp)]
    df = df[df['Temperature'] <= int(max_temp)]
    return df


def check_pres_range(df, min_pres, max_pres):
    '''find pressure within a user range from slider'''
    df = df[df['Pressure'] >= int(min_pres)]
    df = df[df['Pressure'] <= int(max_pres)]
    return df


def rename(df):
    '''remove underscores in particle properties in classification column'''
    rename_types = dict(zip(globals.particle_types, globals.particle_types_rename))
    df = df.replace(rename_types)
    return df


def update_layout(fig, df, contour=False):
    fig.update_layout(
        {
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },
        xaxis_showgrid=True,
        xaxis_zeroline=False,
        title={
            'text': f"n={len(df)}",
            'x': 0.43,
            'xanchor': 'center',
            'yanchor': 'top',
        },
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    if contour:
        return fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    else:
        return fig.update_traces(width=1, points=False)
