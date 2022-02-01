'''CPI campaign data processing functions for dataframes'''
import pandas as pd
import numpy as np
import globals
from dash_extensions.enrich import Output, Input, ServersideOutput


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
        fig.update_layout(legend=dict(itemsizing='constant'))
        return fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    else:
        return fig.update_traces(width=1, points=False)


def register(app):
    @app.callback(
        ServersideOutput("store-df", "data"),
        Input("campaign-dropdown", "value"),
        Input("min-temp", "value"),
        Input("max-temp", "value"),
        Input("min-pres", "value"),
        Input("max-pres", "value"),
        Input("date-picker", 'start_date'),
        Input("date-picker", 'end_date'),
    )
    def preprocess(
        campaign, min_temp, max_temp, min_pres, max_pres, start_date, end_date
    ):
        df = read_campaign(campaign)
        df = remove_bad_props(df)
        df = remove_bad_env(df)
        # print(df['Pressure'].min(), df['Pressure'].max())
        df = rename(df)
        df = check_temp_range(df, min_temp, max_temp)
        df = check_pres_range(df, min_pres[0], max_pres[0])
        df = check_date_range(df, start_date, end_date)
        # tic = datetime.datetime.now()
        # orjson.dumps(df.to_dict(orient='records'))
        # toc = datetime.datetime.now()
        # print(f"TIME TO SERIALIZE = {(toc-tic).total_seconds()}")
        return df

    @app.callback(
        Output('date-picker', 'min_date_allowed'),
        Output('date-picker', 'max_date_allowed'),
        [Input('campaign-dropdown', 'value')],
    )
    def set_date_picker(campaign):
        '''update date picker based on campaign start and end dates'''
        return (
            globals.campaign_start_dates[campaign],
            globals.campaign_end_dates[campaign],
        )
