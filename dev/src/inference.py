import math
import glob
import pickle
import os
import numpy as np
import cv2
import time
from tqdm import tqdm

import constants as c
import alarm

from bmog import BMOG
from roi_extractor import regions_of_interest

import matplotlib.pyplot as plt

from libs.efficientnet.efficientnet.tfkeras import preprocess_input
import efn


def imread_scaled(image_path, scale):
    image = cv2.imread(image_path)
    w,h,_ = image.shape
    w,h = int(w*scale), int(h*scale)
    return cv2.resize(image, (h,w))


def predict(image):
    global model
    img_shape = model.input_shape[1:3]

    image = cv2.resize(image, img_shape)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return model.predict(image)[0][0]


def inference_on_frame(image, out_dir, image_scale=1.0):
    global bgs
    global bgs_params
    global classification_threshold
    global stop_prediction

    DISPLAY = True # display bgs+predictions on screen (orange=tractor, blue=other)
    SAVE = False # save image on hdd
    ALARM = True # send alarm if tractor detected

    # for displaying with matplotlib
    # if DISPLAY:
    #     fig,axs = plt.subplots(1,2)
    #     fig.subplots_adjust(hspace=0, wspace=0)
    #     axs[0].set_xticks([]), axs[0].set_yticks([])
    #     axs[1].set_xticks([]), axs[1].set_yticks([])
    #     axs[0].set_title("Background Subtraction")
    #     axs[1].set_title("Predictions")


    # do backgroundsubtraction
    fg_mask = bgs.apply(image)

    # dilate fgmask to fuse close objects
    if bgs_params['dilations'] > 0:
        fg_mask = cv2.dilate(fg_mask, (5,5), iterations=bgs_params['dilations'])

    # search rois
    rois = regions_of_interest(fg_mask,
            min_size=bgs_params['min_size'],
            max_size=c.max_size*image_scale,
            aspect_ratio=c.aspect_ratio)

    fg_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

    # classify rois
    predictions = []
    for roi in rois:
        x,y,x2,y2 = roi
        crop = image[y:y2, x:x2]
        prediction = predict(crop)
        predictions.append(prediction)

        if DISPLAY or SAVE:
            color = (255-255*prediction, 127, 255*prediction)
            thickness = int(6*image_scale)

            if prediction >= classification_threshold:
                cv2.rectangle(image, (x,y), (x2,y2), color, thickness=thickness)
            cv2.rectangle(fg_color, (x,y), (x2,y2), color, thickness=thickness)

    # send alarm to poweroff wind
    if ALARM:
        did_alarm = alarm.alarm_logic_max(predictions)
        if did_alarm:
            image = cv2.putText(image, 'ALARM', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)

    h, w = image.shape[:2]
    vis = np.zeros((h, 2*w, 3), np.uint8)
    # vis[:h, :w, :3] = cv2.cvtColor(fg_color, cv2.COLOR_GRAY2BGR)
    vis[:h, :w, :3] = fg_color
    vis[:h, w:2*w, :3] = image

    if SAVE and os.path.exists(out_dir):
        fname = f"{out_dir}/{os.path.basename(image_path)}"
        cv2.imwrite(fname, vis)

    if DISPLAY:
        # axs[0].imshow(fg_mask, cmap='gray')
        # axs[1].imshow(fg_color)
        # plt.draw()
        # plt.pause(0.001)
        cv2.namedWindow("Predictions", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Predictions", 1600, 800)
        cv2.imshow("Predictions", vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_prediction = True


def inference_on_video(video_path='0', image_scale=1.0):
    global stop_prediction

    # open video capture
    cap = cv2.VideoCapture(video_path, cv2.CAP_GSTREAMER)

    flag, frame = cap.read()
    while(flag and not stop_prediction):
        inference_on_frame(frame, "", image_scale=image_scale)
        flag, frame = cap.read()
    cv2.destroyAllWindows()
    cap.release()


def inference_on_image_sequence(image_dir, out_dir, image_scale=1.0):
    global stop_prediction

    for image_path in tqdm(sorted(glob.glob(image_dir+'/*.jpg'))):
        image = imread_scaled(image_path, image_scale)
        inference_on_frame(image, out_dir, image_scale=image_scale)

        if stop_prediction:
            break


if __name__ == "__main__":
    # build cnn
    model = efn.build_model(phi=-15, dropout=0.05)

    # load already trained cnn weights
    # if not found, see ../output/README.md
    model_checkpoints = "../output/efn_scale/-15-dropout_0.05-weighted/checkpoints/*.hdf5"
    model.load_weights(max(glob.iglob(model_checkpoints), key=os.path.getctime))
    model.summary()

    # parameters
    classification_threshold = 0.665
    bgs_params = {'threshold': 20, 'postprocessing_size': 7,
            'dilations': 7, 'min_size': 7}

    # initialise background subtractor
    bgs = BMOG(threshold_l=bgs_params['threshold'],
            postprocessing_size=bgs_params['postprocessing_size'])

    # stop if pressed 'q'
    stop_prediction = False

    # ------ live-video -----
    video_path = 0 
    gst_str = ('v4l2src device=/dev/video{} ! '
            'video/x-raw, width=(int){}, height=(int){} ! '
            'videoconvert ! appsink').format(0, 1600, 1200, 2)
    # start inference on video
    inference_on_video(video_path=gst_str, image_scale=0.5)

    # ------- recordings -----
    image_dir_parent = "/home/moritz/Downloads/test_sequences/"
    #image_dir_parent = c.test_sequences_dir
    #sequence = "01_day-rain"
    sequence = "02_night-fog"
    image_dir = f"{image_dir_parent}/{sequence}"

    out_dir = f"{c.output_dir}/{sequence}"
    #inference_on_image_sequence(image_dir, out_dir, image_scale=0.5)
