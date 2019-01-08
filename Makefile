#NCSDKv2 used by default
WRAPPER_FILES := ./wrapper/ncs_wrapper.cpp

#Uncomment the following line to use NCSDKv1
#WRAPPER_FILES := ./wrapper/fp16.c ./wrapper/ncs_wrapper_v1.cpp

#Use rpi_switch.h as config, setup raspicam lib
USE_RPI := $(shell cat rpi_switch.h | sed -n 's:\#define USE_RASPICAM \([0,1]\):\1:p')
ifeq ($(USE_RPI), 0)
	RPI_LIBS := 
	RPI_ARCH := 
else
	RPI_LIBS := -lraspicam
	RPI_ARCH := -march=armv7-a 
endif

OPENVINO_PATH := /opt/intel/computer_vision_sdk
OPENVINO_PATH_RPI := /home/pi/Work/libs/inference_engine_vpu_arm

all: graph_ssd demo_ssd

#switch to desktop/raspberry mode (just edit rpi_switch.h)
switch_desk:
	sed -i 's:\#define USE_RASPICAM 1:\#define USE_RASPICAM 0:' rpi_switch.h
switch_rpi:
	sed -i 's:\#define USE_RASPICAM 0:\#define USE_RASPICAM 1:' rpi_switch.h
convert_yolo:
	# face model is from: https://github.com/dannyblueliu/YOLO-version-2-Face-detection
	#1) enter folder with models
	#2) launch docker image with dl-converter, connect data directories in docker and host,
	#   change user to current user to prevent creating files with root permissions,
	#   run bash->python->converter on specified models 
	#   (.cfg .weights) -> (.prototxt .caffemodel)
	#3) fix input format in .prototxt with utility script
	#4) go back
	cd models/face; \
	sudo docker run -v `pwd`:/workspace/data \
	-u `id -u` -ti dlconverter:latest bash -c \
	"python ./pytorch-caffe-darknet-convert/darknet2caffe.py \
	./data/yolo-face.cfg ./data/yolo-face_final.weights \
	./data/yolo-face.prototxt ./data/yolo-face.caffemodel"; \
	python ../../utils/fix_proto_input_format.py yolo-face.prototxt yolo-face-fix.prototxt; \
	cd ../..
graph_yolo:
	cd models/face; \
	mvNCCompile -s 12 -o graph -w yolo-face.caffemodel yolo-face-fix.prototxt; \
	cd ../..
graph_ssd:
	cd models/face; \
	mvNCCompile -s 12 -o graph_ssd -w ssd-face.caffemodel ssd-face.prototxt; \
	cd ../..
graph_ssd_longrange:
	cd models/face; \
	mvNCCompile -s 12 -o graph_ssd -w ssd-face-longrange.caffemodel ssd-face-longrange.prototxt; \
	cd ../..
demo_yolo:
	g++ \
	-I/usr/include -I. \
	-L/usr/lib/x86_64-linux-gnu \
	-L/usr/local/lib \
	yolo.cpp detection_layer.c $(WRAPPER_FILES) \
	-o demo \
	-lmvnc $(RPI_LIBS) \
	`pkg-config opencv --cflags --libs` 
demo_ssd:
	g++ \
	-I/usr/include -I. \
	-L/usr/lib/x86_64-linux-gnu \
	-L/usr/local/lib \
	ssd.cpp $(WRAPPER_FILES) \
	-o demo \
	-lmvnc $(RPI_LIBS) \
	`pkg-config opencv --cflags --libs`
model_vino:
	cp $(OPENVINO_PATH)/deployment_tools/intel_models/face-detection-retail-0004/FP16/face-detection-retail-0004.bin \
	./models/face/vino.bin; \
	cp $(OPENVINO_PATH)/deployment_tools/intel_models/face-detection-retail-0004/FP16/face-detection-retail-0004.xml \
	./models/face/vino.xml
model_vino_big:
	cp $(OPENVINO_PATH)/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.bin \
	./models/face/vino.bin; \
	cp $(OPENVINO_PATH)/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml \
	./models/face/vino.xml
model_vino_custom:
	mo.py \
	--framework caffe \
	--input_proto models/face/ssd-face.prototxt \
	--input_model models/face/ssd-face.caffemodel \
	--output_dir models/face \
	--model_name ssd-vino-custom \
	--mean_values [127.5,127.5,127.5] \
	--scale_values [127.5,127.5,127.5] \
	--data_type FP16; \
	cp ./models/face/ssd-vino-custom.xml ./models/face/vino.xml; \
	cp ./models/face/ssd-vino-custom.bin ./models/face/vino.bin
model_vino_rpi:
	wget --no-check-certificate \
	https://download.01.org/openvinotoolkit/2018_R4/open_model_zoo/face-detection-retail-0004/FP16/face-detection-retail-0004.xml \
	-O ./models/face/vino.xml; \
	wget --no-check-certificate \
	https://download.01.org/openvinotoolkit/2018_R4/open_model_zoo/face-detection-retail-0004/FP16/face-detection-retail-0004.bin \
	-O ./models/face/vino.bin
model_vino_big_rpi:
	wget --no-check-certificate \
	https://download.01.org/openvinotoolkit/2018_R4/open_model_zoo/face-detection-adas-0001/FP16/face-detection-adas-0001.xml \
	-O ./models/face/vino.xml; \
	wget --no-check-certificate \
	https://download.01.org/openvinotoolkit/2018_R4/open_model_zoo/face-detection-adas-0001/FP16/face-detection-adas-0001.bin \
	-O ./models/face/vino.bin
model_vino_custom_rpi:
	cp ./models/face/ssd-vino-custom.xml ./models/face/vino.xml; \
	cp ./models/face/ssd-vino-custom.bin ./models/face/vino.bin
demo_vino: 
	g++ $(RPI_ARCH) \
	-I/usr/include -I. \
	-I$(OPENVINO_PATH)/deployment_tools/inference_engine/include \
	-I$(OPENVINO_PATH_RPI)/deployment_tools/inference_engine/include \
	-L/usr/lib/x86_64-linux-gnu \
	-L/usr/local/lib \
	-L$(OPENVINO_PATH)/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64 \
	-L$(OPENVINO_PATH_RPI)/deployment_tools/inference_engine/lib/raspbian_9/armv7l \
	vino.cpp wrapper/vino_wrapper.cpp \
	-o demo -std=c++11 \
	`pkg-config opencv --cflags --libs` \
	-ldl -linference_engine $(RPI_LIBS)
profile_yolo: convert_yolo
	cd models/face; \
	mvNCProfile yolo-face-fix.prototxt -w yolo-face.caffemodel -s 12; \
	cd ../..
