# Face detection demo for Movidius Neural Compute Stick

### NOTE
This project was upgraded to use NCSDK v2, which is not backward-compatible. Files for compiling with NCSDK v1 are also provided: you will have to edit `#include <./wrapper/ncs_wrapper.hpp>` in `yolo.cpp` and `ssd.cpp`, as well as `WRAPPER_FILES` variable in `Makefile`.

## Intro

This project is a C++ face detection demo for NCS, with different models: 
* <a href="https://github.com/BeloborodovDS/MobilenetSSDFace" target="_blank"> My Mobilenet-SSD models</a>
* <a href="https://github.com/dannyblueliu/YOLO-version-2-Face-detection" target="_blank"> YOLO v2 model </a> 


To build and run this demo <a href="https://developer.movidius.com/start" target="_blank">NCSDK</a> and OpenCV are needed.

| Model 		| FPS (Core i3) |
|---			|---		|
|Mobilenet-SSD		|10.4		|
|Mobilenet-SSD longrange|11.4		|
|YOLO v2		|5.1		|

## Mobilenet-SSD

To run Mobilenet-SSD face detection demo:
~~~
make graph_ssd
make demo_ssd
./demo
~~~

or simply
~~~
make
./demo
~~~

Or with long-range face detector (pruned, a bit faster, for small faces only):
~~~
make graph_ssd_longrange
make demo_ssd
./demo
~~~

## YOLO v2

Model is converted from Darknet to Caffe format with <a href="https://github.com/marvis/pytorch-caffe-darknet-convert" target="_blank">Pytorch\Caffe\Darknet converter</a> in a Docker container and then compiled for NCS. You can skip Docker conversion if you download converted model from my drive.

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
make convert_yolo
make graph_yolo
make demo_yolo
./demo
~~~

Alternatively (skip conversion from Darknet to Caffe):

1. Download converted .caffemodel from <a href="https://drive.google.com/open?id=17PgRAkMLrhFORCEqefdZEHKoPHXmduZJ" target="_blank">my Drive</a> and place it in `models/face`.

2. Compile graph to NCS and build demo:
~~~
make graph_yolo
make demo_yolo
./demo
~~~

