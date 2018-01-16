all: graph demo
graph:
	cd models/empty; \
	mvNCCompile -s 12 -o graph -w empty.caffemodel empty.prototxt; \
	cd ../..
demo:
	g++ \
	-I/usr/include \
	-L/usr/lib/x86_64-linux-gnu \
	-L/usr/local/lib \
	test.cpp fp16.c \
	-o test \
	-lopencv_core -lopencv_imgproc -lopencv_highgui \
	-lopencv_video \
	-lmvnc \
	`pkg-config opencv --cflags --libs` 

