#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <ctime>

//Neural compute stick
#include <mvnc.h>
#include "./fp16.h"
#include "./detection_layer.h"

using namespace std;
using namespace cv;

#define NETWORK_INPUT_SIZE 448

//read whole graph file into buffer, return pointer to buffer, set filesize
void* readGraph(const char* filename, int* filesize)
{
    ifstream file(filename, ios::binary);
    
    file.seekg (0, file.end);
    *filesize = file.tellg();
    file.seekg (0, file.beg);

    char* buffer = new char[*filesize];
    if (file.read(buffer, *filesize))
    {
	cout<<"File size: "<<(*filesize)<<endl;
	file.close();
	return (void*)buffer;
    }
    
    file.close();
    return NULL;
}


int main()
{
    mvncStatus ncsCode;
    void *ncsDevice;
    char ncsName[100];
    
    //Get NCS name
    ncsCode = mvncGetDeviceName(0, ncsName, 100);
    if (ncsCode != MVNC_OK)
    {
        cout<<"Cannot find NCS device, status: "<<ncsCode<<endl;
        return 0;
    }
    cout<<"Found device named "<<ncsName<<endl;
    
    //Open NCS device via its name
    ncsCode = mvncOpenDevice(ncsName, &ncsDevice);
    if (ncsCode != MVNC_OK)
    {  
        cout<<"Cannot open NCS device, status: "<<ncsCode<<endl;
        return 0;
    }
  
    //Get graph file size and data
    int graphSize=0;
    void* graphData=NULL;
    graphData = readGraph("./models/face/graph", &graphSize);
    
    if (graphData==NULL)
    {
      cout<<"Cannot open graph file\n";
      return 0;
    }
    
    //Allocate computational graph
    void* ncsGraph;
    ncsCode = mvncAllocateGraph(ncsDevice, &ncsGraph, graphData, graphSize);
    if (ncsCode != MVNC_OK)
    {
        cout<<"Cannot allocate graph, status: "<<ncsCode<<endl;
	delete [] (char*)graphData;
	return 0;
    }
  
    //Init camera from OpenCV
    VideoCapture cap;
    if(!cap.open(0))
    {
        cout<<"Cannot open camera!"<<endl;
        return 0;
    }
    
    //variables for getting result from NCS
    unsigned int resultSize;
    void* result16f;
    void* otherParam;
    
    //Capture-Render cycle
    clock_t start;
    int nframes=0;
    start = clock();
    for(;;)
    {
        nframes++;
      
	//Get frame
	Mat frame;
        cap >> frame;
	
	//transform frame
	if (frame.channels()==4)
	  cvtColor(frame, frame, CV_BGRA2BGR);
	resize(frame, frame, Size(NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE));
	flip(frame, frame, 1);
	frame.convertTo(frame, CV_32F, 1/255.0);
        
	//transform to 16f
	Mat frame16f(NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE, CV_16UC3);
	floattofp16((unsigned char*)frame16f.data, (float*)frame.data, frame.rows*frame.cols*3);
	
	//load image to NCS
	ncsCode = mvncLoadTensor(ncsGraph, (void*)frame16f.data, frame16f.rows*frame16f.cols*3*sizeof(unsigned short), NULL);
        if (ncsCode != MVNC_OK)
        {
            cout<<"Cannot load image to NCS, status: "<<ncsCode<<endl;
	    break;
        }
        
        //get and decode result from NCS
	ncsCode = mvncGetResult(ncsGraph, &result16f, &resultSize, &otherParam);
        if (ncsCode != MVNC_OK)
	{
	    cout<<"Cannot retrieve result from NCS, status: "<<ncsCode<<endl;
	    break;
	}
	int nres = resultSize/sizeof(unsigned short);
	float* result = new float[nres];
	fp16tofloat(result, (unsigned char*)result16f, nres);
	
	//get boxes and probs and draw them
	vector<Rect> rects;
	vector<float> probs;
	get_detection_boxes(result, NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE, 0.2, probs, rects);
	for (int i=0; i<rects.size(); i++)
	{
	    if (probs[i]>0)
	      rectangle(frame, rects[i], Scalar(0,0,255,probs[i]));
	}
	
	delete [] result;
	
	imshow("render", frame);
	
        //Exit if any key pressed
        if (waitKey(1)!=-1)
        {
            break;
        }
    }
    
    //calculate fps
    double time;
    time = (clock()-start)/(double)CLOCKS_PER_SEC;
    cout<<"Frame rate: "<<nframes/time<<endl;
    
    waitKey(0);
    
    ncsCode = mvncDeallocateGraph(ncsGraph);
    ncsGraph = NULL;
    
    ncsCode = mvncCloseDevice(ncsDevice);
    ncsDevice = NULL;
    if (ncsCode != MVNC_OK)
    {
        cout<<"Cannot close NCS device, status: "<<ncsCode<<endl;
    }
    
    //clear graph data
    delete [] (char*)graphData;
    
}
