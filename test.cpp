#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <ctime>

//Neural compute stick
#include <mvnc.h>
#include "./detection_layer.h"

//to use NCSDKv1, replace this file by ncs_wrapper_v1.hpp
#include <./wrapper/ncs_wrapper.hpp>

using namespace std;
using namespace cv;

#define NETWORK_INPUT_SIZE  448
#define NETWORK_OUTPUT_SIZE 1331

int main()
{
    //NCS interface
    NCSWrapper NCS(NETWORK_INPUT_SIZE*NETWORK_INPUT_SIZE*3, NETWORK_OUTPUT_SIZE);
    
    //Start communication with NCS
    if (!NCS.load_file("./models/face/graph"))
      return 0;
  
    //Init camera from OpenCV
    VideoCapture cap;
    if(!cap.open(0))
    {
        cout<<"Cannot open camera!"<<endl;
        return 0;
    }
    
    Mat frame;
    Mat resized(NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE, CV_8UC3);
    Mat resized16f(NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE, CV_32FC3);
    float* result;
    
    //Capture-Render cycle
    int nframes=0;
    int64 start = getTickCount();
    for(;;)
    {
        nframes++;
      
	//Get frame
        cap >> frame;
	
	//transform frame
	if (frame.channels()==4)
	  cvtColor(frame, frame, CV_BGRA2BGR);
	flip(frame, frame, 1);
	resize(frame, resized, Size(NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE));
	cvtColor(resized, resized, CV_BGR2RGB);
	resized.convertTo(resized16f, CV_32F, 1/255.0);
        
	if(!NCS.load_tensor((float*)resized16f.data, result))
	{
	  NCS.print_error_code();
	  break;
	}
	
	//get boxes and probs
	vector<Rect> rects;
	vector<float> probs;
	get_detection_boxes(result, frame.cols, frame.rows, 0.2, probs, rects);
	
	//non-maximum suppression
	do_nms(rects, probs, 1, 0.4);
	
	//draw boxes and render frame
	for (int i=0; i<rects.size(); i++)
	{
	    if (probs[i]>0) 
	      rectangle(frame, rects[i], Scalar(0,0,255));
	}
	imshow("render", frame);
	
        //Exit if any key pressed
        if (waitKey(1)!=-1)
        {
            break;
        }
    }
    
    //calculate fps
    double time = (getTickCount()-start)/getTickFrequency();
    cout<<"Frame rate: "<<nframes/time<<endl;
    
}
