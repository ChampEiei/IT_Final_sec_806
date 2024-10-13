import pandas as pd
import plotly.figure_factory as ff
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px

# Load the data
i7 = pd.read_excel('https://raw.githubusercontent.com/ChampEiei/it_group_806/main/colleration.xlsx')
mse = pd.read_excel('https://raw.githubusercontent.com/ChampEiei/it_group_806/main/MSE_table.xlsx')
df_val = pd.read_excel('https://raw.githubusercontent.com/ChampEiei/it_group_806/main/df_val.xlsx')
df_val.index = df_val['Unnamed: 0']
df_pred = pd.read_excel('https://raw.githubusercontent.com/ChampEiei/it_group_806/main/df_pred.xlsx')
df_pred.index = df_pred['Unnamed: 0']
i7.rename(columns={'act_gdp': 'GDP','มูลค่า (ส่งออก-ตัวมันเอง)':'Export','มูลค่า(สกุล บาท)':'Import'}, inplace=True)
df_pred['class']  = df_pred['class'].apply(lambda  x : x.replace('✅',''))
df_val['class']  = df_val['class'].apply(lambda  x : x.replace('✅',''))
mse['calss']=mse['calss'].apply(lambda x :x.replace('✅',''))
# Calculate correlation matrix
selected_columns = ['GDP', 'set100', 'Export', 'Import']
i7 = i7[selected_columns]
correlation_matrix = i7.corr().values
columns = selected_columns

# สร้าง annotation text ที่มีทศนิยม 2 ตำแหน่ง
annotation_text = [[f"{val:.2f}" for val in row] for row in correlation_matrix]

# สร้าง heatmap ด้วย plotly เพื่อแสดง correlation matrix
correlation_fig = ff.create_annotated_heatmap(z=correlation_matrix, x=columns, y=columns,
                                              annotation_text=annotation_text, colorscale='Blues', showscale=True)
correlation_fig.update_layout(
    title_text='Correlation Matrix (Plotly)',
    plot_bgcolor='#1a1a2e',
    paper_bgcolor='#1a1a2e',
    font=dict(color='#ffffff'),
    xaxis=dict(color='#ffffff'),
    yaxis=dict(color='#ffffff')
)

# Load GDP data
df_gdp = pd.read_excel("https://raw.githubusercontent.com/ChampEiei/IT_Final_sec_806/main/gdp_real.xlsx")
df1_gdp = pd.read_excel("https://raw.githubusercontent.com/ChampEiei/IT_Final_sec_806/main/gdp_newewst.xlsx")
df1_gdp.rename(columns={'act_gdp': 'GDP'}, inplace=True)
df1_gdp['class'] = 'augment'
df_gdp['class'] = 'actual'
gdp = pd.concat([df1_gdp, df_gdp], ignore_index=True)

# Load SET100 data
set100=pd.read_csv("https://raw.githubusercontent.com/ChampEiei/IT_Final_sec_806/main/set100_final.xls")
# Load Import data
df_im = pd.read_excel("https://raw.githubusercontent.com/yanisasomnam/Prayfah/cf4cdfbea9bb903a433d38d43b58370f032656e7/imbeforegroup1.xlsx")
df_im['Year-Month'] = pd.to_datetime(df_im['ปี-เดือน'], format='%Y-%m-%d')
a = df_im[['ประเภท', 'HS 2dg']]
b = a.groupby(['HS 2dg'])['ประเภท'].first().reset_index()
df_final_im = df_im.groupby(['Year-Month', 'HS 2dg'])['มูลค่า(สกุล บาท)'].sum().reset_index()
df_final_im = pd.merge(df_final_im, b, on='HS 2dg', how='left')

# Load Export data
df_export_grouped = pd.read_csv('https://raw.githubusercontent.com/ChampEiei/it_group_806/main/export_dash.xls')

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1(children='Prediction LSTM GDP Dashboard', style={
        'textAlign': 'center',
        'color': '#ffffff',
        'background': 'linear-gradient(90deg, #1a1a2e, #16213e)',
        'padding': '10px',
        'fontWeight': 'bold'
    }),

    html.Div(children='Prediction Vs Actual.', style={
        'textAlign': 'center',
        'color': '#ecf0f1',
        'background': 'linear-gradient(90deg, #1a1a2e, #16213e)',
        'padding': '10px',
        'fontSize': '20px',
    }),

    # Dropdown for data filtering
    html.Div(
        children=[
        dcc.Dropdown(
            id='type-filter',
            options=[{'label': i, 'value': i} for i in df_val['class'].unique()],
            value='GDP Export mech',
            multi=False,
            style={
                'backgroundColor': '#1E90FF'
            }
        )
    ],
    style={'padding': '10px', 'backgroundColor': '#34495e'}
    ),

    # Graphs for scatter and line
    html.Div(
        children=[
            html.Div(children=[dcc.Graph(id='scatter-graph')], style={
                'width': '48%',
                'display': 'inline-block',
                'padding': '10px',
                'background': 'linear-gradient(90deg, #16213e, #0f3460)',
                'border': '2px solid #ffffff',  # กรอบขาวรอบกราฟ
                'color': '#ffffff'
            }),
            html.Div(children=[dcc.Graph(id='line')], style={
                'width': '45%',
                'display': 'inline-block',
                'padding': '10px',
                'background': 'linear-gradient(90deg, #16213e, #0f3460)',
                'border': '2px solid #ffffff',  # กรอบขาวรอบกราฟ
                'color': '#ffffff'
            }),
        ],
        style={'display': 'flex', 'justify-content': 'space-between', 'flex-wrap': 'wrap', 'padding': '10px', 'background': 'linear-gradient(90deg, #1a1a2e, #16213e)'}
    ),

    # Heatmap and Bar Graph
    html.Div([
        html.Div([dcc.Graph(id='heatmap', figure=correlation_fig)], style={
            'width': '48%',
            'display': 'inline-block',
            'padding': '10px',
            'background': 'linear-gradient(90deg, #16213e, #0f3460)',
            'border': '2px solid #ffffff',  # กรอบขาวรอบกราฟ
        }),
        html.Div([dcc.Graph(id='MSE_graph', figure={'data': [go.Bar(x=mse['calss'], y=mse['MSE'], orientation='v')], 'layout': go.Layout(title='MSE for Each Loop', plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e', font=dict(color='#ffffff'))})], style={
            'width': '44%',
            'display': 'inline-block',
            'padding': '10px',
            'background': 'linear-gradient(90deg, #16213e, #0f3460)',
            'border': '2px solid #ffffff',  # กรอบขาวรอบกราฟ
        })
    ], style={'display': 'flex', 'justify-content': 'space-between', 'flex-wrap': 'wrap', 'padding': '10px', 'background': 'linear-gradient(90deg, #1a1a2e, #16213e)'}),

    html.H1("Economic Dashboard", style={
        'background': 'linear-gradient(90deg, #1a1a2e, #16213e)',
        'padding': '10px',
        'color': '#ecf0f1',
        'textAlign': 'center'
    }),

    dcc.DatePickerRange(id='date-picker-range', start_date=gdp['Date'].min(), end_date=gdp['Date'].max(), display_format='YYYY-MM-DD', style={'backgroundColor': '#2c3e50', 'color': '#ffffff'}),

    html.Div([
        html.Div([
            dcc.Dropdown(id='class-dropdown-gdp', options=[{'label': cls, 'value': cls} for cls in gdp['class'].unique()], value=[cls for cls in gdp['class'].unique()], multi=True, style={'backgroundColor': '#34495e'}),
            dcc.Graph(id='gdp-graph'),
            html.P(id='gdp-value', style={'fontWeight': 'bold', 'color': '#ecf0f1'})
        ], style={
            'width': '45%',
            'display': 'inline-block',
            'background': 'linear-gradient(90deg, #16213e, #0f3460)',
            'border': '2px solid #ffffff',  # กรอบขาวรอบกราฟ
            'padding': '10px'
        }),

        html.Div([
            dcc.Dropdown(id='class-dropdown-set100', options=[{'label': cls, 'value': cls} for cls in set100['class'].unique()], value=[cls for cls in set100['class'].unique()], multi=True, style={'backgroundColor': '#34495e'}),
            dcc.Graph(id='set100-graph')
        ], style={
            'width': '45%',
            'display': 'inline-block',
            'float': 'right',
            'background': 'linear-gradient(90deg, #16213e, #0f3460)',
            'border': '2px solid #ffffff',  # กรอบขาวรอบกราฟ
            'padding': '10px'
        }),
    ],style={'background': 'linear-gradient(90deg, #16213e, #0f3460)'}),

    html.Div([
        html.Div([
            dcc.Dropdown(id='class-dropdown-import', options=[cls for cls in df_final_im['ประเภท'].unique()], value=[cls for cls in df_final_im['ประเภท'].unique()], multi=True, style={'backgroundColor': '#34495e'}),
            dcc.Graph(id='import-graph'),
            html.P(id='import-value', style={'fontWeight': 'bold', 'color': '#ecf0f1'})
        ], style={
            'width': '45%',
            'display': 'inline-block',
            'background': 'linear-gradient(90deg, #16213e, #0f3460)',
            'border': '2px solid #ffffff',  # กรอบขาวรอบกราฟ
            'padding': '10px'
        }),

        html.Div([
            dcc.Dropdown(id='class-dropdown-export', options=[{'label': cls, 'value': cls} for cls in df_export_grouped['ประเภท'].unique()], value=[cls for cls in df_export_grouped['ประเภท'].unique()], multi=True, style={'backgroundColor': '#34495e'}),
            dcc.Graph(id='export-graph'),
            html.P(id='export-value', style={'fontWeight': 'bold', 'color': '#ecf0f1'})
        ], style={
            'width': '45%',
            'display': 'inline-block',
            'float': 'right',
            'background': 'linear-gradient(90deg, #16213e, #0f3460)',
            'border': '2px solid #ffffff',  # กรอบขาวรอบกราฟ
            'padding': '10px'
        }),
    ],style={'background': 'linear-gradient(90deg, #16213e, #0f3460)'}),

    ##new
    html.Div([
        html.Div([

            dcc.Graph(id='import-graph_pie')
        ], style={
            'width': '45%',
            'display': 'inline-block',
            'background': 'linear-gradient(90deg, #16213e, #0f3460)',
            'border': '2px solid #ffffff',  # กรอบขาวรอบกราฟ
            'padding': '10px'
        }),

        html.Div([

            dcc.Graph(id='export-graph_pie')
        ], style={
            'width': '45%',
            'display': 'inline-block',
            'float': 'right',
            'background': 'linear-gradient(90deg, #16213e, #0f3460)',
            'border': '2px solid #ffffff',  # กรอบขาวรอบกราฟ
            'padding': '10px'
        }),
    ],style={'background': 'linear-gradient(90deg, #16213e, #0f3460)'})
],style={'background': 'linear-gradient(90deg, #16213e, #0f3460)'})


# Define the callbacks to update graphs based on filter
@app.callback(
    [Output('scatter-graph', 'figure'), Output('line', 'figure')],
    [Input('type-filter', 'value')]
)
def update_graphs(selected_pl_type):
    if selected_pl_type:
        filtered_df_all = df_pred[df_pred['class'] == selected_pl_type]
        fill_val = df_val[df_val['class'] == selected_pl_type]
    else:
        filtered_df_all = df_pred[df_pred['class'] == 'gdp,set100']
        fill_val = df_val[df_val['class'] == 'gdp,set100']

    scatter_fig = go.Figure(data=[
        go.Scatter(x=filtered_df_all.index, y=filtered_df_all['Predict'], mode='lines', name='Predicted'),
        go.Scatter(x=filtered_df_all.index, y=filtered_df_all['trainY'], mode='lines', name='Actual')
    ])

    scatter_fig.update_layout(
        title='Predicted Vs Actual',
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#ffffff'),
        xaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
        yaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
        legend=dict(font=dict(color='#ffffff'))
    )

    line_fig = go.Figure(data=[
        go.Scatter(x=fill_val.index, y=fill_val['loss'], mode='lines', name='loss'),
        go.Scatter(x=fill_val.index, y=fill_val['val_loss'], mode='lines', name='val_loss')
    ])

    line_fig.update_layout(
        title='Validation And Train Loss',
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#ffffff'),
        xaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
        yaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
        legend=dict(font=dict(color='#ffffff'))
    )

    return scatter_fig, line_fig

@app.callback(
    Output('gdp-graph', 'figure'),
    Output('gdp-value', 'children'),
    Input('class-dropdown-gdp', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_gdp_graph(selected_classes, start_date, end_date):
    if selected_classes:

        filtered_df = gdp[gdp['class'].isin(selected_classes) & (gdp['Date'] >= start_date) & (gdp['Date'] <= end_date)]
    else:
        filtered_df = gdp.copy()
    fig = px.line(filtered_df, x='Date', y='GDP', color='class', title='GDP Over Time')

    fig.update_layout(
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#ffffff'),
        xaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
        yaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
        legend=dict(font=dict(color='#ffffff'))
    )

    gdp_sum = filtered_df['GDP'].sum()
    return fig, f"Total GDP: {gdp_sum}"

@app.callback(
    Output('set100-graph', 'figure'),
    [Input('class-dropdown-set100', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_set100_graph(selected_classes, start_date, end_date):
    if selected_classes:

        filtered_df = set100[set100['class'].isin(selected_classes) &
                             (set100['Month-Year'] >= start_date) &
                             (set100['Month-Year'] <= end_date)]
    else :
        filtered_df = set100.copy()


    fig = px.line(filtered_df, x='Month-Year', y='SET1003/', color='class', title='SET100 Over Time')

    fig.update_layout(
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#ffffff'),
        xaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
        yaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
        legend=dict(font=dict(color='#ffffff'))
    )

    return fig

@app.callback(
    Output('import-graph', 'figure'),
    Output('import-value', 'children'),
    Input('class-dropdown-import', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_import_graph(selected_hs2dg, start_date, end_date):
    if selected_hs2dg:

        filtered_df = df_final_im[df_final_im['ประเภท'].isin(selected_hs2dg) &
                                  (df_final_im['Year-Month'] >= start_date) &
                                  (df_final_im['Year-Month'] <= end_date)]
    else:
        filtered_df = df_final_im.copy()

    fig = px.scatter(filtered_df, x='Year-Month', y='มูลค่า(สกุล บาท)', color='ประเภท', title='Import Over Time')

    fig.update_layout(
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#ffffff'),
        xaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
        yaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
        legend=dict(font=dict(color='#ffffff'))
    )

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
    if selected_hs2dg:

        filtered_df = df_export_grouped[df_export_grouped['ประเภท'].isin(selected_hs2dg) &
                                        (df_export_grouped['Year-Month'] >= start_date) &
                                        (df_export_grouped['Year-Month'] <= end_date)]
    else :
        filtered_df= df_export_grouped.copy()
    fig = px.scatter(filtered_df, x='Year-Month', y='มูลค่า (บาท)', color='ประเภท', title='Export Over Time')

    fig.update_layout(
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#1a1a2e',
            font=dict(color='#ffffff'),
            xaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
            yaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
            legend=dict(font=dict(color='#ffffff'))
    )

    export_sum = filtered_df['มูลค่า (บาท)'].sum()
    return fig, f"Total Export Value: {export_sum:,.2f} บาท"

@app.callback(
    Output('export-graph_pie', 'figure'),
   Input('class-dropdown-export', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'))
def update_export_graph_pie(selected_hs2dg, start_date, end_date):
    if selected_hs2dg:

        filtered_df = df_export_grouped[df_export_grouped['ประเภท'].isin(selected_hs2dg) &
                                        (df_export_grouped['Year-Month'] >= start_date) &
                                        (df_export_grouped['Year-Month'] <= end_date)]
    else :
        filtered_df= df_export_grouped.copy()
    fig = px.pie(filtered_df, names='ประเภท', values='มูลค่า (บาท)', title='Export Over Time')
    fig.update_layout(
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#1a1a2e',
            font=dict(color='#ffffff'),
            xaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
            yaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
            legend=dict(font=dict(color='#ffffff'))
    )
    return fig

@app.callback(
    Output('import-graph_pie', 'figure'),
   Input('class-dropdown-import', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'))
def update_import_graph_pie(selected_hs2dg, start_date, end_date):

    if selected_hs2dg:

        filtered_df = df_final_im[df_final_im['ประเภท'].isin(selected_hs2dg) &
                                        (df_final_im['Year-Month'] >= start_date) &
                                        (df_final_im['Year-Month'] <= end_date)]
    else :
        filtered_df= df_final_im.copy()
    fig = px.pie(filtered_df, names='ประเภท', values='มูลค่า(สกุล บาท)', title='Import Over Time')
    fig.update_layout(
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#1a1a2e',
            font=dict(color='#ffffff'),
            xaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
            yaxis=dict(showgrid=False, zeroline=False, color='#ffffff'),
            legend=dict(font=dict(color='#ffffff'))
    )
    return fig
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True,port=1111)
