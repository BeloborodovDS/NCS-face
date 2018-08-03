# Face detection demo for Movidius Neural Compute Stick

This project is a C++ face detection demo for NCS, with <a href="https://github.com/dannyblueliu/YOLO-version-2-Face-detection" target="_blank">YOLO2 model</a>.

Model is converted from Darknet to Caffe format with <a href="https://github.com/marvis/pytorch-caffe-darknet-convert" target="_blank">Pytorch\Caffe\Darknet converter</a> and then compiled for NCS.

To build and run this demo <a href="https://developer.movidius.com/start" target="_blank">NCSDK</a>, Docker and OpenCV are needed.

1. Download model (.cfg and .weights) from <a href="https://github.com/dannyblueliu/YOLO-version-2-Face-detection" target="_blank">here</a> into `models/face`.

2. Compile Docker image with converter:
~~~
cd utils
sudo docker build -t dlconverter .
~~~
(this is just a hack, you can try to clone converter repo and run the script instead)

3. build and run demo
~~~
make
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
