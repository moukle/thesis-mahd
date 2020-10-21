import libs.efficientnet.efficientnet.tfkeras as efn

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

import numpy as np

def build_model(phi, dropout):
    w = round(1.1 ** phi, 2)
    d = round(1.2 ** phi, 2)
    r = int(round(1.15 ** phi, 2) * 224)

    model_base = efn.EfficientNet(
        w, d, r, dropout,
        model_name=f'efn-{phi}',
        weights=None,
        include_top=False)

    x = model_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout, name='top_dropout')(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=model_base.input, outputs=x)

    return model