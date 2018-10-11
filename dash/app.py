import cv2 as cv
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import dash_reusable_components as drc
import dct
import mseplot

external_stylesheets = [
    # Normalize the CSS
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    # Fonts
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto"
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    # Custom CSS
    "https://cdn.rawgit.com/xhlulu/dash-image-processing/1d2ec55e/custom_styles.css",
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.scripts.config.serve_locally = True

app.layout = html.Div([
    # Banner display
    html.Div([
        html.H2(
            'Image Compression App',
            id='title'
        ),
        html.Img(
            src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"
        )
    ], className="banner"),

    dcc.Upload(
        id='upload-image',
        children=[
            'Drag and Drop or ',
            html.A('Select Files')
        ],
        style={
            'height': 'auto',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '10px',
            'textAlign': 'center',
            'margin': '10px',
            'cursor': 'pointer',
        },
        accept='image/*'
    ),
    html.Div([
        html.Button("convert to grayscale", id="button-gray"),

        dcc.Dropdown(placeholder="Select compression method",
                     id="dropdown",
                     options=[
                         {'label': 'Quantization Table', 'value': 'scale_factor'},
                         {'label': 'No. of DCT coefficients', 'value': 'coeff'},
                     ]),

        dcc.Input(placeholder="Enter Factor...",
                  id="input-box",
                  type='number'),

        html.Button("compress image", id="button-compress")

    ], className='button-container'),

    html.Div(children=[
        html.Div(id='div-original-image', className='six columns',
                 children=[
                     dcc.Graph(id='original-image', style={'height': '80vh'})
                 ]),
        html.Div(id='div-compressed-image', className='six columns',
                 children=[
                     dcc.Graph(id='compressed-image', style={'height': '80vh'})
                 ])
    ]),

    html.Button('plot graph', id='button-plot', style={'display': 'block', 'margin': '5px auto'}),
    html.Div(id='plot-graph')
], className='container')


@app.callback(Output('div-original-image', 'children'),
              [Input('button-gray', 'n_clicks'),
               Input('upload-image', 'contents')])
def update_image(n_clicks, content):
    img = drc.base64_to_cv(content)
    if n_clicks is not None:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return [
        drc.InteractiveImage(
            image_id='original-image',
            image=img,
        )]


@app.callback(Output('div-compressed-image', 'children'),
              [Input('button-compress', 'n_clicks')],
              [State('upload-image', 'contents'),
               State('dropdown', 'value'),
               State('input-box', 'value')])
def update_compressedimage(n_clicks, content, mode, factor):
    factor = float(factor)
    img = drc.base64_to_cv(content)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if mode == 'scale_factor':
        cimg = dct.compress(img, scale_factor=factor)
    else:
        factor = int(factor)
        cimg = dct.compress(img, num_coeffs=factor)

    return [
        drc.InteractiveImage(
            image_id='compressed-image',
            image=cimg,
        )]


@app.callback(Output('plot-graph', 'children'),
              [Input('button-plot', 'n_clicks')],
              [State('upload-image', 'contents'),
               State('dropdown', 'value')])
def update_graph(n_clicks, content, mode):
    img = drc.base64_to_cv(content)
    return [mseplot.mseplot(img, mode)]


if __name__ == '__main__':
    app.run_server(debug=True, port=8060)
