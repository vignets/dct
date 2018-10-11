import base64

import cv2 as cv
import dash_core_components as dcc
import dash_html_components as html
import numpy as np

# Variables
HTML_IMG_SRC_PARAMETERS = 'data:image/jpeg;base64,'


# Display utility functions
def _merge(a, b):
    return dict(a, **b)


def _omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}


# Image utility functions
def cv_to_base64(img):
    # Convert captured image to JPG
    retval, buffer = cv.imencode('.jpg', img)

    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text.decode()


def base64_to_cv(uri):
    if uri is not None:
        encoded_data = uri.split(',')[-1]

        r = base64.b64decode(encoded_data)
        nparr = np.frombuffer(r, dtype=np.uint8)
        # nparr = np.fromstring(encoded_data.decode('base64'), np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        return img


# Custom Display Components
def Card(children, **kwargs):
    return html.Section(
        children,
        style=_merge({
            'padding': 5,
            'margin': 5,
            'borderRadius': 5,
            # 'height': 'auto',
            "overflow": 'auto',
            'border': 'thin lightgrey solid',

            # Remove possibility to select the text for better UX
            'user-select': 'none',
            '-moz-user-select': 'none',
            '-webkit-user-select': 'none',
            '-ms-user-select': 'none'
        }, kwargs.get('style', {})),
        **_omit(['style'], kwargs)
    )


# Custom Image Components
def InteractiveImage(image_id, image):
    encoded_image = cv_to_base64(image)

    height, width, *channel = image.shape

    return dcc.Graph(
        id=image_id,
        figure={
            'data': [{
                'x': [0, width],
                'y': [0, height],
                'mode': 'markers',
                'marker': {'opacity': 0}}],
            'layout': {
                # 'margin': go.layout.Margin(l=40, b=40, t=26, r=10),
                'xaxis': {
                    'range': (0, width),
                    'scaleanchor': 'y',
                    'showgrid': False
                },
                'yaxis': {
                    'range': (0, height),
                    'showgrid': False
                },
                'images': [{
                    'xref': 'x',
                    'yref': 'y',
                    'x': 0,
                    'y': 0,
                    'yanchor': 'bottom',
                    'sizing': 'stretch',
                    'sizex': width,
                    'sizey': height,
                    'layer': 'below',
                    'source': HTML_IMG_SRC_PARAMETERS + encoded_image,
                }],
            }
        },
        style={
            'width': '100%',
            'height': '80vh'
        },

        config={
            'modeBarButtonsToRemove': [
                'sendDataToCloud',
                'toggleSpikelines',
                'hoverClosestCartesian',
                'hoverCompareCartesian',
            ]
        },
    )
