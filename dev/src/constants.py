import os

# IMAGE DIRECTORIES
DATA_DIR_BASE= "../data"

sequences_dir = f"{DATA_DIR_BASE}/sequences_images"
rois_dir = f"{DATA_DIR_BASE}/sequences_rois"
rois_sorted_dir = f"{DATA_DIR_BASE}/sequences_rois_sorted"

bgs_sequences_dir = f"{DATA_DIR_BASE}/bgs_val_sequences"
bgs_annotation_dir = f"{DATA_DIR_BASE}/bgs_val_annotations"

test_sequences_dir = f"{DATA_DIR_BASE}/test_sequences"
test_annotation_dir = f"{DATA_DIR_BASE}/test_annotations"


# BGS
frames_for_bgs_init = 5

# ROI EXTRACTION
min_size = 15
max_size = 1000
aspect_ratio = 0.2

# CNN
train_dir = f"{DATA_DIR_BASE}/sequences_rois_split/unsampled/train/"
val_dir = f"{DATA_DIR_BASE}/sequences_rois_split/unsampled/val/"

train_dir_up = f"{DATA_DIR_BASE}/sequences_rois_split/upsampled/train/"

output_dir = "../output"
output_dir_bgs = f"{output_dir}/bmog"
output_dir_models = f"{output_dir}/efn"
output_dir_inference = f"{output_dir}/inference"


batch_size = 32
epochs = 100
epochs_lr = 30

max_lrs = {
    -6 : 0.1669,
    -20: 0.1669, 
    -25: 0.1669,
    -30: 0.1669}
# OTHER
class_labels = {0: "other", 1: "agriculture"}
labels = "other", "agriculture"