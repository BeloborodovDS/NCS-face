# Collection of face detection demos for Movidius Neural Compute Stick 

Desktop / Raspberry Pi

NCSDK / NCSDK2 / OpenVINO

### NOTE
This project was upgraded to use NCSDK v2, which is not backward-compatible. Files for compiling with NCSDK v1 are also provided: you will have to edit `#include <./wrapper/ncs_wrapper.hpp>` in `yolo.cpp` and `ssd.cpp`, as well as `WRAPPER_FILES` variable in `Makefile`.

## Intro

This project is a C++ face detection demo for NCS (both desktop Ubuntu and Raspberry Pi with Raspbian Stretch), with different models: 
* <a href="https://github.com/BeloborodovDS/MobilenetSSDFace" target="_blank"> My Mobilenet-SSD detector</a>
* <a href="https://github.com/BeloborodovDS/MobilenetSSDFace" target="_blank"> My Mobilenet-SSD longrange detector</a> (a bit faster, only for small faces)
* <a href="https://github.com/dannyblueliu/YOLO-version-2-Face-detection" target="_blank"> YOLO v2 model from this repo </a> (converted from Darknet to Caffe) 
* Two face detectors from <a href="https://software.intel.com/en-us/OpenVINO-toolkit" target="_blank"> Intel OpenVINO</a>, namely face-detection-retail-0004 and face-detection-adas-0001

To build and run NCSDK demo, <a href="https://github.com/movidius/ncsdk" target="_blank">NCSDK 2</a> and OpenCV are needed.

For OpenVINO demos, <a href="https://software.intel.com/en-us/OpenVINO-toolkit" target="_blank"> Intel OpenVINO SDK</a> is needed. 

### Performance (FPS)

| Model 			 	| USB-3 | USB-2 | Raspberry Pi |
|---				 	|---	 |---	 |---		|
|Custom SSD with NCSDK2		 	|10.8	 |9.3	 |7.2		|
|Custom longrange SSD with NCSDK2	|11.8	 |10.0	 |7.3		|
|YOLO v2 with NCSDK2		 	|5.3	 |4.6	 |3.6		|
|Custom SSD with OpenVINO	 	|10.6	 |9.9	 |7.9		|
|OpenVINO face-detection-retail-0004	|15.6	 |14.2	 |9.3		|
|OpenVINO face-detection-adas-0001	|5.8	 |5.5	 |3.9		|

* "USB-3" setting: Core i7 / Ubuntu 16 VM / Neural Compute Stick at USB-3 
* "USB-2" setting: Core i7 / Ubuntu 16 VM / Neural Compute Stick with USB-2 cord at USB-3 (so effective communication is USB-2)
* "Raspberry Pi" setting: Raspberry Pi 2B / Raspbian Stretch / Neural Compute Stick with USB-2 cord at USB-2 

## Compiling for desktop or Raspberry Pi

I use <a href="http://www.uco.es/investiga/grupos/ava/node/40" target="_blank"> Raspicam library </a> to access Raspberry Pi camera. You will have to install it to run this demo on Raspberry Pi.
This code was tested on Raspberry Pi 2 model B, however I believe it should work on other models too.

File `rpi_switch.h` contains a single variable to switch between desktop and RPi modes. 
Makefile contains two targets to set this variable: use `make switch_desk` to switch to desktop mode (default) and `make switch_rpi` to switch to Raspberry mode (you can run them just once).

Targets `convert_yolo, graph_yolo, graph_ssd, graph_ssd_longrange` will not work on Raspberry. You will have to compile graph files on your Ubuntu desktop and copy them to your RPi.
For convenience, two compiled graph files are provided (YOLO and full SSD). If they do not work, then you probably have different NCSDK version, so you will have to recompile them.

NOTE: YOLO detector is quite sensitive to light conditions, and I failed to get good results on Raspberry. SSD works fine, however.

## Custom Mobilenet-SSD with NCSDK2

To run custom Mobilenet-SSD face detection demo (desktop):
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

For Raspberry Pi:
After running `make graph_ssd` or `make graph_ssd_longrange` on a desktop machine, copy your `graph_ssd` file to `models/face` on Raspberry (or just use provided file) and run:
~~~
make switch_rpi
make demo_ssd
./demo
~~~ 

## Running detectors with OpenVINO

First, choose a model:
~~~
make model_vino_custom
~~~
to compile custom (my) Mobilenet-SSD detector to OpenVINO format, or

~~~
make model_vino
~~~
to copy face-detection-retail-0004 detector from OpenVINO installation folder, or

~~~
make model_vino_big
~~~
to copy face-detection-adas-0001 detector from OpenVINO installation folder.

Then compile and run the demo:
~~~
make demo_vino
./demo
~~~

Do not forget to update OpenVINO installation path in Makefile, if necessary. 
Also, you can switch between models without recompiling the demo.

To run on Raspberry Pi, use model targets with "rpi" suffix: model_vino_rpi, model_vino_big_rpi 
(downloaded instead of being copied, since there are no models in Raspbian OpenVINO distribution) 
and model_vino_custom_rpi (not converted, just copied from inside current repo, since mo.py is also unavailable).

Then switch, compile and run the demo:
~~~
make switch_rpi
make demo_vino
./demo
~~~

## YOLO v2 with NCSDK2

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

For Raspberry Pi:
After running `make graph_yolo` on a desktop machine, copy your `graph` file to `models/face` on Raspberry (or just use provided file) and run:
~~~
make switch_rpi
make demo_yolo
./demo
~~~ 

