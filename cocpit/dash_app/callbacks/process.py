'''CPI campaign data processing functions for dataframes'''

import pandas as pd
import globals
import numpy as np
from dash_extensions.enrich import Input, Output, State, ServersideOutput, dcc
import dash


# def read_campaign(campaign):
#     '''read particle property df and environmental property df_env
#     merged based on filename and date'''

#     campaign = 'CRYSTAL_FACE_NASA' if campaign == 'CRYSTAL FACE (NASA)' else campaign
#     campaign = 'CRYSTAL_FACE_UND' if campaign == 'CRYSTAL FACE (UND)' else campaign
#     campaign = 'ICE_L' if campaign == 'ICE L' else campaign
#     campaign = 'AIRS_II' if campaign == 'AIRS II' else campaign
#     df = pd.read_parquet(
#         f"../../final_databases/vgg16/v1.4.0/merged_env/{campaign}.parquet",
#         engine='fastparquet',
#     )

#     return df


# def remove_bad_data(df):
#     '''remove missing or bad environmental data'''
#     df = df.replace([-999.99, -999.0, np.inf, -np.inf], np.nan).dropna()
#     df = df[
#         (df['Latitude'] != 0)
#         & (df['Longitude'] != 0)
#         & (df['Pressure'] != 0)
#         & (df['Ice Water Content'] > 1e-5)
#         & (df["Complexity"] != -0.0)
#         & (df['Particle Height'] != 0.0)
#         & (df['Particle Width'] != 0.0)
#     ]
#     return df


# def rename(df):
#     '''remove underscores in particle properties in classification column'''
#     rename_types = dict(zip(globals.particle_types, globals.particle_types_rename))
#     df = df.replace(rename_types)
#     rename_types = dict(zip(globals.campaigns, globals.campaigns_rename))
#     df = df.replace(rename_types)
#     return df


# def update_layout(fig, contour=False, margin=20, height=300):
#     '''update figures to have white background, and include and center sample size in title'''
#     fig.update_layout(
#         {
#             'plot_bgcolor': 'rgba(0, 0, 0, 0)',
#             'paper_bgcolor': 'rgba(0, 0, 0, 0)',
#         },
#         margin=dict(l=margin, r=margin, t=margin, b=margin),
#         xaxis_showgrid=True,
#         xaxis_zeroline=False,
#         showlegend=False,
#         height=height,
#     )

#     fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
#     if contour:
#         fig.update_layout(legend=dict(itemsizing='constant'))
#         return fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
#     else:
#         return fig.update_traces(width=1, points=False)


def register(app):
    #     @app.callback(
    #         [
    #             ServersideOutput("df-classification", "data"),
    #             ServersideOutput("df-lat", "data"),
    #             ServersideOutput("df-lon", "data"),
    #             ServersideOutput("df-alt", "data"),
    #             ServersideOutput("df-date", "data"),
    #             ServersideOutput("df-prop", "data"),
    #             ServersideOutput("df-env", "data"),
    #             ServersideOutput("df-temp", "data"),
    #             ServersideOutput("len-df", "data"),
    #             ServersideOutput("store-df", "data"),
    #         ],
    #         [
    #             Input('submit-button', 'n_clicks'),
    #             Input('top-down-map', 'selectedData'),
    #             State("campaign-dropdown", "value"),
    #             State("part-type-dropdown", "value"),
    #             State("min-temp", "value"),
    #             State("max-temp", "value"),
    #             State("min-pres", "value"),
    #             State("max-pres", "value"),
    #             State("date-picker", 'start_date'),
    #             State("date-picker", 'end_date'),
    #             State("min-size", "value"),
    #             State("max-size", "value"),
    #             State("property-dropdown", "value"),
    #             State("env-dropdown", "value"),
    #         ],
    #         memoize=True,
    #     )
    #     def preprocess(
    #         nclicks,
    #         selected_data,
    #         campaign,
    #         part_type,
    #         min_temp,
    #         max_temp,
    #         min_pres,
    #         max_pres,
    #         start_date,
    #         end_date,
    #         min_size,
    #         max_size,
    #         part_prop,
    #         env_prop,
    #     ):
    #         '''read campaign data and process based on user input from menu'''

    #         df = read_campaign(campaign)
    #         df = rename(df)
    #         df = remove_bad_data(df)
    #         df = df[df['Classification'].isin(part_type)]
    #         df['max_dim'] = np.maximum(df['Particle Width'], df['Particle Height'])
    #         df['min_dim'] = np.minimum(df['Particle Width'], df['Particle Height'])
    #         df = df[(df['min_dim'] >= int(min_size)) & (df['max_dim'] <= int(max_size))]
    #         df['date'] = df['date'].str.split(' ').str[0]
    #         df = df[df['date'].between(start_date, end_date)]
    #         df = df[df['Temperature'].between(int(min_temp), int(max_temp))]
    #         df = df[df['Pressure'].between(int(min_pres[0]), int(max_pres[0]))]

    #         # hone data to lats and lons chosen by box select
    #         if selected_data and selected_data['points']:
    #             sel_data = pd.DataFrame(selected_data['points'])
    #             df = df[
    #                 (df['Longitude'] < sel_data['lon'].max())
    #                 & (df['Longitude'] > sel_data['lon'].min())
    #                 & (df['Latitude'] > sel_data['lat'].min())
    #                 & (df['Latitude'] < sel_data['lat'].max())
    #             ]

    #         return (
    #             df['Classification'],
    #             df['Latitude'],
    #             df['Longitude'],
    #             df['Altitude'],
    #             df['date'],
    #             df[part_prop],
    #             df[env_prop],
    #             df['Temperature'],
    #             len(df),
    #             df,
    #         )

    #     @app.callback(
    #         Output('image-count', 'children'),
    #         Input('df-classification', 'data'),
    #     )
    #     def change_image_count(df_classification):
    #         '''update image count card based on number of points in map selection'''
    #         return len(df_classification)

    #     @app.callback(
    #         Output("download-df-csv", "data"),
    #         [Input("download-button", "n_clicks"), State('store-df', 'data')],
    #         prevent_initial_call=True,
    #     )
    #     def func(n_clicks, df):
    #         return dcc.send_data_frame(df.to_csv, "cocpit.csv")

    #     @app.callback(
    #         [
    #             Output('date-picker', 'min_date_allowed'),
    #             Output('date-picker', 'max_date_allowed'),
    #             Output('date-picker', 'start_date'),
    #             Output('date-picker', 'end_date'),
    #         ],
    #         Input('campaign-dropdown', 'value'),
    #     )
    #     def set_date_picker(campaign):
    #         '''update date picker based on campaign start and end dates'''
    #         return (
    #             globals.min_dates[campaign],
    #             globals.max_dates[campaign],
    #             globals.campaign_start_dates[campaign],
    #             globals.campaign_end_dates[campaign],
    #         )

    #     @app.callback(
    #         Output('flight-count', 'children'),
    #         [
    #             Input('df-date', 'data'),
    #         ],
    #     )
    #     def find_iops(df_date):
    #         '''update number of flights after filtering dates'''

    #         df_date = pd.to_datetime(df_date)
    #         grouped_df = df_date.groupby([df_date.dt.month, df_date.dt.day])
    #         return grouped_df.ngroups

    @app.callback(
        Output(f"navbar-collapse", "is_open"),
        [Input(f"navbar-toggler", "n_clicks")],
        [State(f"navbar-collapse", "is_open")],
    )
    def toggle_navbar_collapse(n, is_open):
        if n:
            return not is_open
        return is_open
