# Base Image
FROM tensorflow/tensorflow:2.0.0-gpu-py3-jupyter


# Core linux dependencies.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
	apt-get install -y \
		git \
		vim \
		software-properties-common

# install most python dependencies via pip
RUN pip install --upgrade pip \
	&& pip install \
		numpy \
		scipy \
		matplotlib \
		scikit-learn \
		pillow \
		jupyterlab \
		xmltodict \
		scikit-image \
		tqdm \
	&& pip install git+https://www.github.com/keras-team/keras-contrib.git

# install opencv-contrib
WORKDIR /
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" \
	&& apt-get install -y \
		wget unzip build-essential \
	&& apt-get install -y \
		cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
	&& apt-get install -y \
		python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev \
    && apt-get install -y \
		yasm libv4l-dev libatlas-base-dev gfortran libpq-dev \
	&& apt-get install -y \
		libharfbuzz-dev \
		vtk7  libvtk7-dev \
		libblas-dev liblapack-dev \
		libxvidcore-dev libx264-dev \
		libgtk-3-dev libatlas-base-dev \
		libglu1-mesa-dev freeglut3-dev mesa-common-dev \
		libeigen3-dev libprotobuf-dev \
	&& apt-get -y clean all \
    && rm -rf /var/lib/apt/lists/*


ENV OPENCV_VERSION="3.4.8"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip \
	&& wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip -O opencv_contrib.zip \
	&& unzip opencv.zip \
	&& unzip opencv_contrib.zip \
	&& mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
	&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
	&& cmake \
		-DBUILD_TIFF=ON \
		-DBUILD_opencv_java=OFF \
		-DWITH_CUDA=OFF \
		-DENABLE_AVX=ON \
		-DWITH_OPENGL=ON \
		-DWITH_OPENCL=ON \
		-DWITH_IPP=ON \
		-DWITH_TBB=ON \
		-DWITH_EIGEN=ON \
		-DWITH_V4L=ON \
		-DBUILD_TESTS=OFF \
		-DBUILD_PERF_TESTS=OFF \
		-DCMAKE_BUILD_TYPE=RELEASE \
		-DOPENCV_GENERATE_PKGCONFIG=ON \
		-DCMAKE_INSTALL_PREFIX=/usr/local \
		-DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
		-DPYTHON_EXECUTABLE=$(which python3) \
		-DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
		-DPYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
		-DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules \
		-DCMAKE_DISABLE_FIND_PACKAGE_TBB=ON \
		.. \
	&& make -j8 \
	&& make install \
	&& rm /opencv.zip \
	&& rm /opencv_contrib.zip \
	&& rm -r /opencv-${OPENCV_VERSION} \
	&& rm -r /opencv_contrib-${OPENCV_VERSION}
