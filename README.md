# Face detection demo for Movidius Neural Compute Stick

### NOTE
This project was upgraded to use NCSDK v2, which is not backward-compatible. Files for compiling with NCSDK v1 are also provided: you will have to edit inclusion of header files in demo .cpp files, as well as WRAPPER_FILES variable in Makefile

This project is a C++ face detection demo for NCS, with <a href="https://github.com/dannyblueliu/YOLO-version-2-Face-detection" target="_blank">YOLO2 model</a>.

Model is converted from Darknet to Caffe format with <a href="https://github.com/marvis/pytorch-caffe-darknet-convert" target="_blank">Pytorch\Caffe\Darknet converter</a> and then compiled for NCS.

To build and run this demo <a href="https://developer.movidius.com/start" target="_blank">NCSDK</a>, Docker and OpenCV are needed.

1. Download model (.cfg and .weights) from <a href="https://github.com/dannyblueliu/YOLO-version-2-Face-detection" target="_blank">here</a> into `models/face`.

2. Compile Docker image with converter:
~~~
cd utils
sudo docker build -t dlconverter .
cd ..
~~~
(this is just a hack, you can try to clone converter repo and run the script instead)

3. build and run demo
~~~
make
./test
~~~

Alternatively (skip convertion from Darknet to Caffe):

1. Download converted .caffemodel from <a href="https://drive.google.com/open?id=17PgRAkMLrhFORCEqefdZEHKoPHXmduZJ" target="_blank">my Drive</a> and place it in models/face.

2. Compile graph to NCS and build demo:
~~~
make graph_yolo
make demo
./test
~~~

## New feature: another demo with Mobilenet-SSD face detector

This detector seems to be better at detection, while 2x faster and 9x smaller.

See my repo for details on this detector, including training: https://github.com/BeloborodovDS/MobilenetSSDFace

To run demo with this detector:
~~~
make graph_ssd
make demo_ssd
./test
~~~

Or long-range face detector (a bit faster, for small faces only):
~~~
make graph_ssd_longrange
make demo_ssd
./test
~~~

