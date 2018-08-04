#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <ctime>
#include <vector>

//Neural compute stick
#include <mvnc.h>
#include <./wrapper/ncs_wrapper.hpp>

using namespace std;
using namespace cv;

#define NETWORK_INPUT_SIZE  300
#define NETWORK_OUTPUT_SIZE 707

void get_detection_boxes(float* predictions, int w, int h, float thresh, 
			 std::vector<float>& probs, std::vector<cv::Rect>& boxes)
{
    int num = predictions[0];
    float score = 0;
    float cls = 0;
    
    for (int i=1; i<num+1; i++)
    {
      score = predictions[i*7+2];
      cls = predictions[i*7+1];
      if (score>thresh && cls<=1)
      {
	probs.push_back(score);
	boxes.push_back(Rect(predictions[i*7+3]*w, predictions[i*7+4]*h,
			    (predictions[i*7+5]-predictions[i*7+3])*w, 
			    (predictions[i*7+6]-predictions[i*7+4])*h));
      }
    }
    
}

int main()
{
    //NCS interface
    NCSWrapper NCS(NETWORK_INPUT_SIZE*NETWORK_INPUT_SIZE*3, NETWORK_OUTPUT_SIZE);
    
    //Start communication with NCS
    if (!NCS.load_file("./models/face/graph_ssd"))
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
    resized16f = Scalar(0);
    cap >> frame; //to get size
    float* result;
    
    //Capture-Render cycle
    int nframes=0;
    int64 start = getTickCount();
    
    vector<Rect> rects;
    vector<float> probs;
    for(;;)
    {
        nframes++;
		  
	//load data to NCS
	if(!NCS.load_tensor_nowait((float*)resized16f.data))
	{
	  if (NCS.ncsCode == MVNC_MYRIAD_ERROR)
	  {
	    char* err;
	    unsigned int len;
	    mvncGetGraphOption(NCS.ncsGraph, MVNC_DEBUG_INFO, (void*)&err, &len);
	    cout<<string(err, len)<<endl;
	  }
	  break;
	}
      
	//draw boxes and render frame
	for (int i=0; i<rects.size(); i++)
	{
	    if (probs[i]>0) 
	      rectangle(frame, rects[i], Scalar(0,0,255));
	}
	imshow("render", frame);
	
	//Get frame
	cap >> frame;
	
	//transform next frame while NCS works
	if (frame.channels()==4)
	  cvtColor(frame, frame, CV_BGRA2BGR);
	flip(frame, frame, 1);
	resize(frame, resized, Size(NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE));
	//cvtColor(resized, resized, CV_BGR2RGB);
	resized.convertTo(resized16f, CV_32F, 1/127.5, -1);
	
	//get result from NCS
	if(!NCS.get_result(result))
	{
	  if (NCS.ncsCode == MVNC_MYRIAD_ERROR)
	  {
	    char* err;
	    unsigned int len;
	    mvncGetGraphOption(NCS.ncsGraph, MVNC_DEBUG_INFO, (void*)&err, &len);
	    cout<<string(err, len)<<endl;
	  }
	  break;
	}
	
	//get boxes and probs
	probs.clear();
	rects.clear();
	get_detection_boxes(result, frame.cols, frame.rows, 0.2, probs, rects);
	
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
