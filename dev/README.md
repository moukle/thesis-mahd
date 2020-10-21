# Instructions
## ML-PC
### SSH
``` bash
git push jarvis master
scp -r contexted_sorted_augmented $USER@jarvis:/home/$USER/data/contexted/sorted_augmented
scp -r sorted $USER@jarvis:/home/$USER/data/sorted
scp -r split $USER@jarvis:/home/$USER/data/split

ssh $USER@jarvis -L 16006:localhost:6006 -L 18888:localhost:8888
```


### Docker
``` bash
# to build and run docker
./docker_build_run.sh

# to attach to running container (docker ps)
docker attach `CONTAINER_ID`
```

### Usage
After running docker via SSH-Tunnel you should be able to connect to `jupyter` by navigating to `localhost:18888` in your browser. If running locally port `:8888`.
Inside `jupyter` you can launch `TensorBoard` in a terminal by executing

``` bash
tensorboard --logdir /tmp/dev/output/efn
```

You can view `TensorBoard` on `localhost:16006` or running locally on port `:6006`.

### TensorBoard
Regex all except _LR_: `^((?!lr).)*$`

---

## Jetson
### Info
Docker image geht nicht, da Baseimage (tensorflow-2.0....) für AMD64 gebaut und nicht ARM64.


### TensorFlow
More information can be found [here](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html).

``` bash
JP_VERSION=42 # check with https://github.com/jetsonhacks/jetsonUtilities

sudo apt-get install python3-pip

sudo pip3 install -U pip testresources setuptools
sudo pip3 install -U numpy grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast h5py astor termcolor protobuf keras-applications keras-preprocessing wrapt google-pasta
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION tensorflow-gpu
# sudo apt install python3-scipy python3-skimage
sudo pip3 install -U pybind11 scikit-image
```

### EfficientNet
Run first jupytercell in `src/2-fit_cnn_model.ipynb` (`jupyter-lab src`) or

``` bash
git clone https://github.com/qubvel/efficientnet.git libs/efficientnet && \
    sed -i '6i\EfficientNet   = inject_tfkeras_modules(model.EfficientNet)' libs/efficientnet/efficientnet/tfkeras.py && \
    sed -i '364 s/require_flatten=include_top/require_flatten=True/' libs/efficientnet/efficientnet/model.py
```

### OpenCV `3.4.8`
``` bash
wget https://raw.githubusercontent.com/jkjung-avt/jetson_nano/master/install_opencv-3.4.8.sh
chmod +x install_opencv-3.4.8.sh
./install_opencv-3.4.8.sh
```

### BMOG
Compile `src/libs/bmog` to shared libary according to `README.md`.
Test with `python3 src/bmog.py`
