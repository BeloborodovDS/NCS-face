all: graph demo
convert:
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
	-u `id -u` -ti dlconverter:latest \
	bash -c "python ./pytorch-caffe-darknet-convert/darknet2caffe.py \
	./data/yolo-face.cfg ./data/yolo-face_final.weights \
	./data/yolo-face.prototxt ./data/yolo-face.caffemodel"; \
	python ../../utils/fix_proto_input_format.py yolo-face.prototxt yolo-face-fix.prototxt; \
	cd ../..
graph: convert
	cd models/face; \
	mvNCCompile -s 12 -o graph -w yolo-face.caffemodel yolo-face-fix.prototxt; \
	cd ../..
demo:
	g++ \
	-I/usr/include \
	-L/usr/lib/x86_64-linux-gnu \
	-L/usr/local/lib \
	test.cpp fp16.c detection_layer.c \
	-o test \
	-lopencv_core -lopencv_imgproc -lopencv_highgui \
	-lopencv_video \
	-lmvnc \
	`pkg-config opencv --cflags --libs` 
profile: convert
	cd models/face; \
	mvNCProfile yolo-face-fix.prototxt -w yolo-face.caffemodel -s 12; \
	cd ../..

