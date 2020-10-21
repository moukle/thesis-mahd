import datetime
import numpy as np
import os
import glob

import constants as c

import tensorflow as tf
from tensorflow.python.framework import graph_io, convert_to_constants
from tensorflow.keras.models import load_model
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt

#from libs.efficientnet.efficientnet.tfkeras import preprocess_input
import efn


# LOAD MODEL AND WEIGHTS

######### DO SAME FOR PHI = -15 / -5 ##########
model = efn.build_model(phi=-15, dropout=0.15)

model_load_dir = f"{c.output_dir}/efn_scale/-15-dropout_0.05-weighted"
model_checkpoints = f"{model_load_dir}/checkpoints/*.hdf5"
model.load_weights(max(glob.iglob(model_checkpoints), key=os.path.getctime))

img_shape = list(model.input_shape)
img_shape[0] = 1 # replace batch 'None'
img_shape = tuple(img_shape)

saved_model_path = os.path.join(model_load_dir, "saved_model")
tr_model_dir = os.path.join(model_load_dir, "trt_model")

# convert to saved_model
tf.saved_model.save(model, saved_model_path)

# convert to trt model
converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_path)
converter.convert()
converter.save(tr_model_dir)

tr_model_loaded = tf.saved_model.load(
    tr_model_dir, tags=[tag_constants.SERVING])
graph_func = tr_model_loaded.signatures[
    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(
    graph_func)
def wrap_func(*args, **kwargs):
    # Assumes frozen_func has one output tensor
    return frozen_func(*args, **kwargs)[0]

output = wrap_func(tf.random.normal(img_shape)).numpy()

# DEFAULT
start = datetime.datetime.now()

for _ in range(10000):
    input_data = np.array(np.random.random_sample(img_shape), dtype=np.float32)
    model.predict(input_data)
    
end = datetime.datetime.now()
diff_tf = end - start

print("Default")
print("Time for 10000 samples: ", diff_tf)
seconds_for_all = int(diff_tf.total_seconds())
x_per_second = round(1 / (seconds_for_all / 10000))
print("Samples per second: ", x_per_second)


# TRT
start = datetime.datetime.now()

for _ in range(10000):
    input_data = np.array(np.random.random_sample(img_shape), dtype=np.float32)
    tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
    output = wrap_func(tensor).numpy()
    
end = datetime.datetime.now()
diff_tr = end - start

print("TRT")
print("Time for 10000 samples: ", diff_tr)
seconds_for_all = int(diff_tr.total_seconds())
x_per_second = round(1 / (seconds_for_all / 10000))
print("Samples per second: ", x_per_second)