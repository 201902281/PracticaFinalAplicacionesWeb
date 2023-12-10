# Importamos las librerias mínimas necesarias
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import logging
from plotly.subplots import make_subplots
from datetime import datetime
import nbimporter
from LoadFDAData import loadInfoFDA
from LoadFinInfo import loadFinInfo
from AgregatedComputation import agregatedComputation
from MLModels import MLModels
import plotly.express as px
import yfinance as yf
from sklearn.linear_model import LinearRegression
from plotly.tools import mpl_to_plotly


    
    
##Load Información FDA
file_path = '../Datos/FDA Aproval.xlsx'
all_sheets_FDA = pd.ExcelFile(file_path).sheet_names
list_FDA_data = loadInfoFDA()
stock_data_output_list = loadFinInfo(list_FDA_data)
final_output_list = agregatedComputation(stock_data_output_list)


app = dash.Dash()

#app.config.suppress_callback_exceptions = True

logging.getLogger('werkzeug').setLevel(logging.INFO)


tab0_content = html.Div([
    html.H4(
        children = ["Datos de 21 empresas que cotizan en el S&P500 y en el Rusell2000 con la fecha de aprobación de cada medicamento e información relativa al tipo de medicamento aprobado."],
        id = "subtitulo0",
        style ={
            "text-align": "justify",
            "display": "block"
        }
    ),
    dcc.RadioItems(
        id='radio_buttons',
        options=[
            {'label': 'Visualización por Tipo de Indice', 'value': 'opcion1'},
            {'label': 'Visualización por Tipo de Medicamento', 'value': 'opcion2'}
        ],
        value='opcion1'  # Opción seleccionada por defecto
    ),
    html.Div(
        dcc.Graph(
            id = "static_FDA_plot",
            style = {"display": "block"}
        ),
        style={'width': '80%', 'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(
            id = "dynamic_FDA_plot",
            style = {"display": "block"}
        ),
        style={'width': '80%', 'display': 'inline-block'}
    ),
])


tab1_content = html.Div([
        html.H4(
            children = [
                "Para cada compañía, se puede visualizar el análisis acerca de cuándo es vinculante un incremento debido a la aprobación de un medicamento y el impacto de la inclusión de dicha variable en un modelo predictivo."
            ],
            id = "subtitulo1",
            style ={
                "text-align": "justify",
                "display": "block"
            }
        ),

        html.Div( 
            children = [
                html.Div( 
                    children = [
                        dcc.Dropdown(
                            options = all_sheets_FDA,
                            placeholder = "Selecciona una raza",
                            value=all_sheets_FDA[0],
                            id = "dropdown_company",
                            style = {
                                "display": "block",
                                "width": "300px",
                                "margin-left": "10px"
                            }
                        ),
                        html.Div(
                            dcc.Graph(
                                id = "dropdown_figure_1",
                                style = {
                                    "display": "none"
                                }
                            ),
                            style={'width': '80%','display': 'inline-block'}                           
                        ),
                        html.P("En la gráfica superior se observa el precio de cotización de cierre de cada compañía anlizada. Las lineas verticales indican momento en el tiempo donde un medicamento fue aprobado por la FDA."),
                        html.Div(
                            dcc.Graph(
                                id = "dropdown_figure_2",
                                style = {
                                    "display": "none"
                                }
                            ),
                            style={'width': '80%','display': 'inline-block'}                           
                        ),
                        html.P("En la gráfica superior se observa un mapa de calor con un ranking de los incrementos en la cotización de la compañía en los días previos y posterior a la emisión de un medicamento."),
                        html.Div(
                            dcc.Graph(
                                id = "dropdown_figure_3",
                                style = {
                                    "display": "none"
                                }
                            ),
                            style={'width': '80%','display': 'inline-block'}                           
                        ),
                        html.P("En la gráfica superior se observa un gráfico de barras con la información agregada del mapa de calor. El día seleccionado determinará en el modelo predictivo el instante donde se considera el efecto de la aprobación del medicamento en la cotización."),
                        html.Div(
                            dcc.Graph(
                                id = "dropdown_figure_4",
                                style = {
                                    "display": "none"
                                }
                            ),
                            style={'width': '80%','display': 'inline-block'}                           
                        ),
                        html.P("En la gráfica superior la construccuón de dos modelos predictivos de regresión linea donde se observa gráficamente el impacto de la inclusión de la variable exógena en las predicciones de los incrementos en la serie."),
                    ], 
                )
            ]
        )
])

tab2_content = html.Div([
            html.Div( 
            children = [
                html.H4(
                    children = [
                    "No hay ninguna fecha previa a la aprobación del medicamento que determine un cambio en la cotización de la compañía. Además, cuando se elimina el efecto de outliers y hay suficientes datos, el efecto de la componente exógena añadida se ve minimizado."
                    ],
                    id = "titulo_cuarta_fila",
                    style ={
                        "text-align": "left",
                        "display": "block"
                    }
                ),

                html.Div(
                    children = [
                        dcc.Checklist(
                            options = ["Selected Date","Value of Selected Date","Value of ML Coefficient"],
                            labelStyle = {
                                'display': 'inline-block',
                                'font-size': "18px",
                                'margin-right': "10px"
                            },
                            id = "checklist_cat",
                            style = {
                                "display": "inline-block",
                            }
                        ),
                        html.Button(
                            children = [
                                "Mostrar"
                            ],
                            id = "boton_cat",
                            title = "Mostrar",
                            n_clicks = 0,
                            style = {
                                'background-color': 'lightgrey',
                                'color': 'steelblue',
                                'height': '35px',
                                'width': '100px',
                                'margin-left': '50px',
                                'border-radius': "5px"
                            }
                        )
                    ],
                    style = {
                        "display": "block"
                    }
                ),
                html.Div(
                    dcc.Graph(
                        id = "scatter_plots",
                        style = {
                            "display" : "none"
                        }
                    ),
                    style={'width': '70%','display': 'inline-block'}
                )
            ],
            id = "cuarta_fila",
        )
])

app.layout = html.Div(
    children= [
        html.H1(
            children = [
                "Análisis Financiero Descriptivo en Empresas Farmaceuticas"
            ],
        id = "titulo",
        style = {
            "text-align": "center",
            "text-decoration": "underline",
            "margin-bottom": "20px",
            "padding-top": "20px",
            "height": "50px"
        }
        ),
        dcc.Tabs([
            dcc.Tab(label='Datos FDA', children=tab0_content, style={'backgroundColor': 'lightblue'},
                    selected_style={'backgroundColor': 'lightblue'}),
            dcc.Tab(label='Panel De Companías', children=tab1_content, style={'backgroundColor': 'lightblue'},
                    selected_style={'backgroundColor': 'lightblue'}),
            dcc.Tab(label='Metricas Agregadas', children=tab2_content, style={'backgroundColor': 'lightblue'},
                    selected_style={'backgroundColor': 'lightblue'}),
    ])
    ],
    style = {
        "font-family": "Arial"
    }
)





@app.callback(
    Output("static_FDA_plot","figure"),
    Output("static_FDA_plot","style"),
    Input("radio_buttons","value")
)
def figure_radio_button(radio_buttons_value):
    
    total_list_dosage = []
    total_list_index = []
    total_list_date = []

    for i in list_FDA_data:
        total_list_dosage = total_list_dosage + list(i["Dosage Form"])
        total_list_index = total_list_index + list(i["Fin Index"])
        total_list_date = total_list_date + list(i["Approval Date"].dt.year)

    total_list_dosage_modified_1 = ["TABLET" if x=="TABLET, EXTENDED RELEASE" else x for x in total_list_dosage]
    total_list_dosage_modified_2 = ["CAPSULE" if x=="CAPSULE, EXTENDED RELEASE" else x for x in total_list_dosage_modified_1]
    contemplated_dosage = ["TABLET","CAPSULE","INJECTABLE","SOLUTION","CREAM","SUSPENSION"]
    total_list_dosage_modified_3 = ["OTHER" if x not in contemplated_dosage else x for x in total_list_dosage_modified_2]
    df_plot_FDA = pd.DataFrame({'Dosage':total_list_dosage_modified_3,'Index': total_list_index,'Year': total_list_date})
    custom_colors = {"TABLET": "green","CAPSULE": "teal","INJECTABLE": "orange","SOLUTION": "grey","CREAM": "purple","SUSPENSION": "brown","OTHER": "blue"}
    if radio_buttons_value == "opcion1":
        static_df_plot_index = df_plot_FDA['Index'].value_counts()
        static_df_plot_index = pd.DataFrame(static_df_plot_index)
        static_df_plot_index = static_df_plot_index.reset_index()
        static_df_plot_index.columns = ['Category','Number of Entries']
        fig = px.bar(static_df_plot_index, y='Category', x='Number of Entries', orientation='h',
                 color='Category',
                 title='Approved Entries by Category')
    else:
        static_df_plot_index = df_plot_FDA['Dosage'].value_counts()
        static_df_plot_index = pd.DataFrame(static_df_plot_index)
        static_df_plot_index = static_df_plot_index.reset_index()
        static_df_plot_index.columns = ['Category','Number of Entries']
        fig = px.bar(static_df_plot_index, y='Category', x='Number of Entries', orientation='h',
                 color='Category',
                 color_discrete_map=custom_colors,
                 title='Approved Entries by Category')
    
    return (fig,{"display":"block"})



@app.callback(
    Output("dynamic_FDA_plot","figure"),
    Output("dynamic_FDA_plot","style"),
    Input("radio_buttons","value")
)
def figure2_radio_button(radio_buttons_value):
    total_list_dosage = []
    total_list_index = []
    total_list_date = []

    for i in list_FDA_data:
        total_list_dosage = total_list_dosage + list(i["Dosage Form"])
        total_list_index = total_list_index + list(i["Fin Index"])
        total_list_date = total_list_date + list(i["Approval Date"].dt.year)

    total_list_dosage_modified_1 = ["TABLET" if x=="TABLET, EXTENDED RELEASE" else x for x in total_list_dosage]
    total_list_dosage_modified_2 = ["CAPSULE" if x=="CAPSULE, EXTENDED RELEASE" else x for x in total_list_dosage_modified_1]
    contemplated_dosage = ["TABLET","CAPSULE","INJECTABLE","SOLUTION","CREAM","SUSPENSION"]
    total_list_dosage_modified_3 = ["OTHER" if x not in contemplated_dosage else x for x in total_list_dosage_modified_2]
    df_plot_FDA = pd.DataFrame({'Dosage':total_list_dosage_modified_3,'Index': total_list_index,'Year': total_list_date})
    
    if radio_buttons_value == "opcion1":
        aggregated_df_index = df_plot_FDA.groupby(['Year', 'Index']).size().reset_index(name='Count')
        aggregated_df_index.columns = ['Year','Category','Count']
        fig = px.bar(aggregated_df_index, x='Year', y='Count', color='Category',
             labels={'Count': 'Number of Entries'},
             title='Approved Entries by Category and by Year')
    else:
        aggregated_df_index = df_plot_FDA.groupby(['Year', 'Dosage']).size().reset_index(name='Count')
        aggregated_df_index.columns = ['Year','Category','Count']
        custom_colors = {"TABLET": "green","CAPSULE": "teal","INJECTABLE": "orange","SOLUTION": "grey","CREAM": "purple","SUSPENSION": "brown","OTHER": "blue"}
        fig = px.bar(aggregated_df_index, x='Year', y='Count', color='Category',
                     color_discrete_map=custom_colors,
                     labels={'Count': 'Number of Entries'},
                     title='Approved Entries by Category and by Year')

    return (fig,{"display":"block"})
    

        


@app.callback(
    Output("dropdown_figure_1", "figure"),
    Output("dropdown_figure_1", "style"),
    Input("dropdown_company", "value"))
def figure_dropdown(dropdown_company_value):
        
    if dropdown_company_value:
        df_plot = stock_data_output_list[all_sheets_FDA.index(dropdown_company_value)]
        fig = px.line(df_plot, x='Date', y='Close', labels={'Date': 'Date', 'Close': 'Closing Price'},
                  title=dropdown_company_value + ": Closing Price Values over Time with Drug Releases")

        # Adding vertical lines for FDA Date with flag equal to 1
        for date, flag in zip(df_plot['Date'], df_plot['FDA Date']):
            if flag == 1:
                fig.add_shape(
                    dict(type='line', x0=date, x1=date, y0=0, y1=1, yref='paper',
                         line=dict(color='grey', width=2, dash='dash'))
                )

        # Setting the x-axis range based on the data
        fig.update_xaxes(range=[df_plot['Date'].min(), df_plot['Date'].max()])
        return (fig,{"display":"block"})
    else:
        return (go.Figure(data = [], layout = {}), {"display": "none"})
    

@app.callback(
    Output("dropdown_figure_2", "figure"),
    Output("dropdown_figure_2", "style"),
    Input("dropdown_company", "value"))
def figure_dropdown_2(dropdown_company_value):
        
    if dropdown_company_value:

        df_plot = stock_data_output_list[all_sheets_FDA.index(dropdown_company_value)]
        matrix_rows = list(df_plot[df_plot["FDA Date"] == 1]["Date"])
        matrix_cols = np.arange(-10, 2)
        perc_change = np.diff(np.array(df_plot["Close"])) / np.array(df_plot["Close"])[:-1] * 100
        perc_change_final = np.insert(perc_change, 0, 0)
        df_output_price = pd.DataFrame(index=range(len(matrix_rows)), columns=range(len(matrix_cols)))


        for j in np.arange(0,len(matrix_rows)):
            date_index = np.where(df_plot['Date'] == matrix_rows[j])
            selected_index = np.arange(date_index[0]-10,date_index[0]+2)
            selected_changes = perc_change_final[selected_index]
            order_of_changes = np.argsort(selected_changes)
            df_output_price.iloc[j] = order_of_changes

    
        matrix_cols = ["Day {}".format(i) for i in matrix_cols]
        df_output_price.columns = matrix_cols
        df_output_price = df_output_price.set_index(pd.Index(matrix_rows))
        average_success = df_output_price.mean()
        selected_date = np.argmin(df_output_price.mean()) - 10
    
        data = {'Day -10': list(df_output_price.iloc[:,0]),
                'Day -9': list(df_output_price.iloc[:,1]),
                'Day -8': list(df_output_price.iloc[:,2]),
                'Day -7': list(df_output_price.iloc[:,3]),
                'Day -6': list(df_output_price.iloc[:,4]),
                'Day -5': list(df_output_price.iloc[:,5]),
                'Day -4': list(df_output_price.iloc[:,6]),
                'Day -3': list(df_output_price.iloc[:,7]),
                'Day -2': list(df_output_price.iloc[:,8]),
                'Day -1': list(df_output_price.iloc[:,9]),
                'Day 0': list(df_output_price.iloc[:,10]),
                'Day 1': list(df_output_price.iloc[:,11])}
        df = pd.DataFrame(data)
        df = df.set_index(pd.Index([timestamp.strftime('%Y-%m-%d') for timestamp in matrix_rows]))
        fig = px.imshow(df, labels=dict(x='Days Since FDA Approval', y='FDA Approval Appointment'),
                    x=df.columns, y=df.index, color_continuous_scale="YlGnBu")
    
        # Adding axis labels and title
        fig.update_layout(
            xaxis_title='Days Since FDA Approval',
            yaxis_title='FDA Approval Appointment Date',
            title_text='Ranking of largest change in the Stock since FDA Approval Appointment'
        )

    
        return (fig,{"display":"block"})
    else:
        return (go.Figure(data = [], layout = {}), {"display": "none"})
    
    
@app.callback(
    Output("dropdown_figure_3", "figure"),
    Output("dropdown_figure_3", "style"),
    Input("dropdown_company", "value"))
def figure_dropdown_3(dropdown_company_value):
        
    if dropdown_company_value:
        df_plot = stock_data_output_list[all_sheets_FDA.index(dropdown_company_value)]
        matrix_rows = list(df_plot[df_plot["FDA Date"] == 1]["Date"])
        matrix_cols = np.arange(-10, 2)
        perc_change = np.diff(np.array(df_plot["Close"])) / np.array(df_plot["Close"])[:-1] * 100
        perc_change_final = np.insert(perc_change, 0, 0)
        df_output_price = pd.DataFrame(index=range(len(matrix_rows)), columns=range(len(matrix_cols)))

        for j in np.arange(0,len(matrix_rows)):
            date_index = np.where(df_plot['Date'] == matrix_rows[j])
            selected_index = np.arange(date_index[0]-10,date_index[0]+2)
            selected_changes = perc_change_final[selected_index]
            order_of_changes = np.argsort(selected_changes)
            df_output_price.iloc[j] = order_of_changes
    
        matrix_cols = ["Day {}".format(i) for i in matrix_cols]
        df_output_price.columns = matrix_cols
        df_output_price = df_output_price.set_index(pd.Index(matrix_rows))
        average_success = df_output_price.mean()
        selected_date = np.argmin(df_output_price.mean()) - 10
        
        Categories = list(df_output_price.columns)
        Average_ranking_position = list(average_success)
        bar_colors = ['darkred' if rank == min(Average_ranking_position) else 'skyblue' for rank in Average_ranking_position]
        df_plot = pd.DataFrame({'Categories': Categories, 'Average_ranking_position': Average_ranking_position, 'Color': bar_colors})
        fig = px.bar(df_plot, x='Categories', y='Average_ranking_position', color='Color',
                 labels={'Categories': 'Date since Approval Appointment', 'Average_ranking_position': 'Average Ranking Position'},
                 title='Average Ranking Position by Date since Approval Appointment',
                 category_orders={'Categories': Categories})

        return (fig,{"display":"block"})
    else:
        return (go.Figure(data = [], layout = {}), {"display": "none"})
    
    
@app.callback(
    Output("dropdown_figure_4", "figure"),
    Output("dropdown_figure_4", "style"),
    Input("dropdown_company", "value"))
def figure_dropdown_4(dropdown_company_value):
        
    if dropdown_company_value:
        ML_output_list = MLModels(all_sheets_FDA.index(dropdown_company_value),stock_data_output_list)
        X_axis = ML_output_list[0]
        Y_real = ML_output_list[1]
        Y_predicted_FDA = ML_output_list[2]
        Y_predicted_WO_FDA = ML_output_list[3]
        
        data = {'X_axis': X_axis,
                'Y_real': Y_real,
                'Y_predicted_FDA': Y_predicted_FDA,
                'Y_predicted_WO_FDA': Y_predicted_WO_FDA}
        df = pd.DataFrame(data)
        df_melted = pd.melt(df, id_vars=['X_axis'], value_vars=['Y_real', 'Y_predicted_FDA', 'Y_predicted_WO_FDA'],
                                var_name='Line', value_name='Return')
        fig = px.line(df_melted, x='X_axis', y='Return', color='Line',
                          labels={'X_axis': 'Time', 'Return': 'Return'},
                          line_shape='linear', line_dash='Line',
                          title='Model Comparison')
        fig.update_layout(
                xaxis_title='Time',
                yaxis_title='Return',
                legend_title_text='Legend')


        return (fig,{"display":"block"})
    else:
        return (go.Figure(data = [], layout = {}), {"display": "none"})


    
    
@app.callback(
    Output("scatter_plots", "figure"),
    Output("scatter_plots", "style"),
    Input("boton_cat", "n_clicks"),
    State("checklist_cat", "value"),
)
def checklist_callback(n_clicks,checklist_cat_value):
    list_index_operated = final_output_list[1]
    list_number_aproved_appointments = final_output_list[2]
    list_key_date = final_output_list[3]
    list_key_date_value_average_ranking = final_output_list[4] 
    list_coeficient = final_output_list[5]
    
    print(checklist_cat_value)
    if (checklist_cat_value is None) or (checklist_cat_value == []):
        return (go.Figure(data = [], layout = {}), {"display":"none"})
    else:
        title_subplot = []
        for i in checklist_cat_value:
            title_subplot.append("Number of Appointments by " + i)
        fig = make_subplots(rows=3, cols=1, subplot_titles=title_subplot)
        col_counter = 0

        for col in checklist_cat_value:
            if col == "Selected Date":
                data = {
                'x': list_number_aproved_appointments,
                'y': list_key_date,
                'text': list_index_operated
                }
            elif col == "Value of Selected Date":
                data = {
                'x': list_number_aproved_appointments,
                'y': list_key_date_value_average_ranking,
                'text': list_index_operated
                }
            else:
                data = {
                'x': list_number_aproved_appointments,
                'y': list_coeficient,
                'text': list_index_operated
                }                
            
            
            df = pd.DataFrame(data)
            col_counter = col_counter + 1
            scatter_trace = go.Scatter(
                x=df['x'],
                y=df['y'],
                mode='markers',
                text=df['text'],
                showlegend=False 
            )
            
            fig['layout']['xaxis'+str(col_counter)]['title']='Number of Aproved Appointments'
            fig['layout']['yaxis'+str(col_counter)]['title']=col
            fig.add_trace(scatter_trace, row=col_counter, col=1)
         
        fig.update_layout()
        
        return (fig,{"display": "block","height":"2000px"})
        

if __name__ == '__main__':
    app.run_server()