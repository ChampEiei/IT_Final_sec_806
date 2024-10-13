import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Load GDP data
df = pd.read_excel("https://raw.githubusercontent.com/ChampEiei/IT_Final_sec_806/main/gdp_real.xlsx")
df1 = pd.read_excel("https://raw.githubusercontent.com/ChampEiei/IT_Final_sec_806/main/gdp_newewst.xlsx")
df1.rename(columns={'act_gdp': 'GDP'}, inplace=True)
df1['class'] = 'augment'
df['class'] = 'actual'
gdp = pd.concat([df1, df], ignore_index=True)

# Load SET100 data
set100real = pd.read_excel("https://raw.githubusercontent.com/ChampEiei/IT_Final_sec_806/main/set100real.xlsx")
set100 = pd.read_excel("https://raw.githubusercontent.com/ChampEiei/IT_Final_sec_806/main/set100.xlsx")
set100real.rename(columns={'วันเดือนปี': 'Month-Year', 'ล่าสุด': 'SET1003/'}, inplace=True)
set100real['Month-Year'] = pd.to_datetime(set100real['Month-Year'], format='%m/%d/%Y').dt.to_period('M').dt.to_timestamp()
set100real['class'] = 'actual'
set100['class'] = 'Groupby'
set100 = pd.concat([set100real, set100], ignore_index=True)

# Load Import data
df_im = pd.read_excel("https://raw.githubusercontent.com/yanisasomnam/Prayfah/cf4cdfbea9bb903a433d38d43b58370f032656e7/imbeforegroup1.xlsx")
df_im['Year-Month'] = pd.to_datetime(df_im['ปี-เดือน'], format='%Y-%m-%d')
a = df_im[['ประเภท', 'HS 2dg']]
b = a.groupby(['HS 2dg'])['ประเภท'].first().reset_index()
df_final_im = df_im.groupby(['Year-Month', 'HS 2dg'])['มูลค่า(สกุล บาท)'].sum().reset_index()
df_final_im = pd.merge(df_final_im, b, on='HS 2dg', how='left')
df_im_grouped = df_im.groupby(['Year-Month', 'HS 2dg'])['มูลค่า(สกุล บาท)'].sum().reset_index()
df_im_grouped = pd.merge(df_im_grouped, b, on='HS 2dg', how='left')

# Example Export data (replace with actual data source)
df_export_grouped = pd.read_csv('https://raw.githubusercontent.com/ChampEiei/it_group_806/main/export_dash.xls')

# Initialize Dash app
app = Dash(__name__)

# Layout for 2x2 dashboard with date slicer and dynamic values
app.layout = html.Div([
    html.H1("Economic Dashboard"),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=gdp['Date'].min(),
        end_date=gdp['Date'].max(),
        display_format='YYYY-MM-DD'
    ),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='class-dropdown-gdp',
                options=[{'label': cls, 'value': cls} for cls in gdp['class'].unique()],
                value=[cls for cls in gdp['class'].unique()],
                multi=True
            ),
            dcc.Graph(id='gdp-graph'),
            html.P(id='gdp-value', style={'fontWeight': 'bold'})
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Dropdown(
                id='class-dropdown-set100',
                options=[{'label': cls, 'value': cls} for cls in set100['class'].unique()],
                value=[cls for cls in set100['class'].unique()],
                multi=True
            ),
            dcc.Graph(id='set100-graph')
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    ]),
    
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='class-dropdown-import',
                options=[cls for cls in df_final_im['ประเภท'].unique()],
                value=[cls for cls in df_final_im['ประเภท'].unique()],
                multi=True
            ),
            dcc.Graph(id='import-graph'),
            html.P(id='import-value', style={'fontWeight': 'bold'})
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Dropdown(
                id='class-dropdown-export',
                options=[{'label': cls, 'value': cls} for cls in df_export_grouped['ประเภท'].unique()],
                value=[cls for cls in df_export_grouped['ประเภท'].unique()],
                multi=True
            ),
            dcc.Graph(id='export-graph'),
            html.P(id='export-value', style={'fontWeight': 'bold'})
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    ])
])

# Callbacks for graphs and dynamic values
@app.callback(
    Output('gdp-graph', 'figure'),
    Output('gdp-value', 'children'),
    Input('class-dropdown-gdp', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_gdp_graph(selected_classes, start_date, end_date):
    filtered_df = gdp[gdp['class'].isin(selected_classes) & (gdp['Date'] >= start_date) & (gdp['Date'] <= end_date)]
    fig = px.line(filtered_df, x='Date', y='GDP', color='class', title='GDP Over Time')
    gdp_sum = filtered_df['GDP'].sum()
    return fig, f"Total GDP: {gdp_sum}"

@app.callback(
    Output('set100-graph', 'figure'),
    Input('class-dropdown-set100', 'value')
)
def update_set100_graph(selected_classes):
    filtered_df = set100[set100['class'].isin(selected_classes)]
    fig = px.line(filtered_df, x='Month-Year', y='SET1003/', color='class', title='SET100 Over Time')
    return fig

@app.callback(
    Output('import-graph', 'figure'),
    Output('import-value', 'children'),
    Input('class-dropdown-import', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_import_graph(selected_hs2dg, start_date, end_date):
    filtered_df = df_final_im[df_final_im['ประเภท'].isin(selected_hs2dg) & 
                              (df_final_im['Year-Month'] >= start_date) & 
                              (df_final_im['Year-Month'] <= end_date)]
    fig = px.scatter(filtered_df, x='Year-Month', y='มูลค่า(สกุล บาท)', color='ประเภท', title='Import Over Time')
    import_sum = filtered_df['มูลค่า(สกุล บาท)'].sum()
    return fig, f"Total Import Value: {import_sum:,.2f} บาท"

@app.callback(
    Output('export-graph', 'figure'),
    Output('export-value', 'children'),
    Input('class-dropdown-export', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_export_graph(selected_hs2dg, start_date, end_date):
    filtered_df = df_export_grouped[df_export_grouped['ประเภท'].isin(selected_hs2dg) & 
                                    (df_export_grouped['Year-Month'] >= start_date) & 
                                    (df_export_grouped['Year-Month'] <= end_date)]
    fig = px.line(filtered_df, x='Year-Month', y='มูลค่า (บาท)', color='ประเภท', title='Export Over Time')
    export_sum = filtered_df['มูลค่า (บาท)'].sum()
    return fig, f"Total Export Value: {export_sum:,.2f} บาท"

if __name__ == '__main__':
    app.run_server(debug=True, port=8055)
