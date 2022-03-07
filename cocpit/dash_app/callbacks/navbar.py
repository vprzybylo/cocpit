from dash_extensions.enrich import Input, Output, State, ServersideOutput, dcc
from callbacks import process
import numpy as np
import pandas as pd
import globals


def register(app):
    @app.callback(
        [
            ServersideOutput("df-classification", "data"),
            ServersideOutput("df-lat", "data"),
            ServersideOutput("df-lon", "data"),
            ServersideOutput("df-alt", "data"),
            ServersideOutput("df-date", "data"),
            ServersideOutput("df-prop", "data"),
            ServersideOutput("df-env", "data"),
            ServersideOutput("df-temp", "data"),
            ServersideOutput("len-df", "data"),
            ServersideOutput("store-df", "data"),
            Output("agg-count", "children"),
            Output("budding-count", "children"),
            Output("bullet-count", "children"),
            Output("column-count", "children"),
            Output("compact-count", "children"),
            Output("planar-count", "children"),
            Output("rimed-count", "children"),
        ],
        [
            Input('submit-button', 'n_clicks'),
            Input('top-down-map', 'selectedData'),
            State("campaign-dropdown", "value"),
            State("part-type-dropdown", "value"),
            State("min-temp", "value"),
            State("max-temp", "value"),
            State("min-pres", "value"),
            State("max-pres", "value"),
            State("date-picker", 'start_date'),
            State("date-picker", 'end_date'),
            State("min-size", "value"),
            State("max-size", "value"),
            State("property-dropdown", "value"),
            State("env-dropdown", "value"),
        ],
        # memoize=True,
    )
    def preprocess(
        nclicks,
        selected_data,
        campaign,
        part_type,
        min_temp,
        max_temp,
        min_pres,
        max_pres,
        start_date,
        end_date,
        min_size,
        max_size,
        part_prop,
        env_prop,
    ):
        '''read campaign data and process based on user input from menu'''

        df = process.read_campaign(campaign)
        df = process.rename(df)
        df = process.remove_bad_data(df)
        df = df[df['Classification'].isin(part_type)]
        df['max_dim'] = np.maximum(df['Particle Width'], df['Particle Height'])
        df['min_dim'] = np.minimum(df['Particle Width'], df['Particle Height'])
        df = df[(df['min_dim'] >= int(min_size)) & (df['max_dim'] <= int(max_size))]
        df = df[df['date'].between(start_date, end_date)]
        df = df[df['Temperature'].between(int(min_temp), int(max_temp))]
        df = df[df['Pressure'].between(int(min_pres[0]), int(max_pres[0]))]

        # hone data to lats and lons chosen by box select
        if selected_data and selected_data['points']:
            sel_data = pd.DataFrame(selected_data['points'])
            df = df[
                (df['Longitude'] < sel_data['lon'].max())
                & (df['Longitude'] > sel_data['lon'].min())
                & (df['Latitude'] > sel_data['lat'].min())
                & (df['Latitude'] < sel_data['lat'].max())
            ]

        agg_count = len(df[df['Classification'] == 'aggregate'])
        budding_count = len(df[df['Classification'] == 'budding rosette'])
        bullet_count = len(df[df['Classification'] == 'bullet rosette'])
        column_count = len(df[df['Classification'] == 'column'])
        compact_count = len(df[df['Classification'] == 'compact irregular'])
        planar_count = len(df[df['Classification'] == 'planar polycrystal'])
        rimed_count = len(df[df['Classification'] == 'rimed'])

        return (
            df['Classification'],
            df['Latitude'],
            df['Longitude'],
            df['Altitude'],
            df['date'],
            df[part_prop],
            df[env_prop],
            df['Temperature'],
            len(df),
            df,
            f'n= {agg_count}',
            f'n= {budding_count}',
            f'n= {bullet_count}',
            f'n= {column_count}',
            f'n= {compact_count}',
            f'n= {planar_count}',
            f'n= {rimed_count}',
        )

    @app.callback(
        Output('image-count', 'children'),
        Input('df-classification', 'data'),
    )
    def change_image_count(df_classification):
        '''update image count card based on number of points in map selection'''
        return len(df_classification)

    @app.callback(
        Output("download-df-csv", "data"),
        [Input("download-button", "n_clicks"), State('store-df', 'data')],
        prevent_initial_call=True,
    )
    def download_data(n_clicks, df):
        return dcc.send_data_frame(df.to_csv, "cocpit.csv")

    @app.callback(
        [
            Output('date-picker', 'min_date_allowed'),
            Output('date-picker', 'max_date_allowed'),
            Output('date-picker', 'start_date'),
            Output('date-picker', 'end_date'),
        ],
        Input('campaign-dropdown', 'value'),
    )
    def set_date_picker(campaign):
        '''update date picker based on campaign start and end dates'''
        return (
            globals.min_dates[campaign],
            globals.max_dates[campaign],
            globals.campaign_start_dates[campaign],
            globals.campaign_end_dates[campaign],
        )

    @app.callback(
        Output('flight-count', 'children'),
        [
            Input('df-date', 'data'),
        ],
    )
    def find_iops(df_date):
        '''update number of flights after filtering dates'''

        df_date = pd.to_datetime(df_date)
        grouped_df = df_date.groupby([df_date.dt.month, df_date.dt.day])
        return grouped_df.ngroups

    @app.callback(
        Output('flight-hours', 'children'),
        [
            Input('df-date', 'data'),
        ],
    )
    def find_flight_hours(df_date):
        '''update number of flight hours after filtering dates'''

        df_date1 = pd.to_datetime(df_date, format='%Y-%m-%d %H:%M:%S')
        group_by_day = df_date1.groupby([df_date1.dt.day])
        sum_hours_over_days = 0
        for day in group_by_day:
            print(day[1].dt.hour.unique())
            sum_hours_over_days += len(day[1].dt.hour.unique())

        return sum_hours_over_days

    @app.callback(
        [
            Output("navbar-collapse", "is_open"),
            Output("navbar-collapse", "style"),
        ],
        [Input("navbar-toggler", "n_clicks")],
        [State("navbar-collapse", "is_open")],
    )
    def toggle_navbar_collapse(n, is_open):
        if n:
            is_open = not is_open
            if is_open is True:
                return (
                    is_open,
                    {
                        'overflow-y': 'scroll',
                        'max-height': '50vh',
                    },
                )

            else:
                return is_open, {}
        else:
            return is_open, {}
