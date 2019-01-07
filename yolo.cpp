#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <ctime>

//Neural compute stick
#include <mvnc.h>
#include "./detection_layer.h"

//to use NCSDKv1, replace this file by ncs_wrapper_v1.hpp
#include <./wrapper/ncs_wrapper.hpp>

#include "./rpi_switch.h"
#if USE_RASPICAM
    #include <raspicam/raspicam.h>
#endif
#define BB_RAW_WIDTH                     1280
#define BB_RAW_HEIGHT                    960

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

#if USE_RASPICAM
    //Init Raspicam camera
    raspicam::RaspiCam Camera;
    Camera.setContrast(100);//100
    Camera.setISO(800);//500
    Camera.setSaturation(-200);//-20
    Camera.setVideoStabilization(true);
    Camera.setExposure(raspicam::RASPICAM_EXPOSURE_ANTISHAKE);
    Camera.setAWB(raspicam::RASPICAM_AWB_AUTO);
    if(!Camera.open())
    {
        cout<<"Cannot open camera with Raspicam!"<<endl;
        return 0;
    }
#else
    //Init camera from OpenCV
    VideoCapture cap;
    if(!cap.open(0))
    {
        cout<<"Cannot open camera with OpenCV!"<<endl;
        return 0;
    }
#endif
    
    Mat frame;
    Mat resized(NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE, CV_8UC3);
    Mat resized16f(NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE, CV_32FC3);
    resized16f = Scalar(0);

    //to get size
#if USE_RASPICAM
    Camera.grab();
    unsigned char* frame_data = Camera.getImageBufferData();
    frame = cv::Mat(BB_RAW_HEIGHT, BB_RAW_WIDTH, CV_8UC3, frame_data);
#else
    cap >> frame; 
#endif

    float* result;
    
    //Capture-Render cycle
    int nframes=0;
    int64 start = getTickCount();
    
    //get boxes and probs
    vector<Rect> rects;
    vector<float> probs;
    for(;;)
    {
        nframes++;
        
        //load data to NCS
        if(!NCS.load_tensor_nowait((float*)resized16f.data))
        {
	    NCS.print_error_code();
	    break;
        }
        
        //draw boxes and render frame
        for (int i=0; i<rects.size(); i++)
        {
            if (probs[i]>0) 
                rectangle(resized, rects[i], Scalar(0,0,255));
        }
        imshow("render", resized);
        
        //Get frame
#if USE_RASPICAM
        Camera.grab();
        frame_data = Camera.getImageBufferData();
        frame = cv::Mat(BB_RAW_HEIGHT, BB_RAW_WIDTH, CV_8UC3, frame_data);
#else
        cap >> frame; 
#endif
        
        //transform frame
        if (frame.channels()==4)
            cvtColor(frame, frame, CV_BGRA2BGR);
        flip(frame, frame, 1);
        resize(frame, resized, Size(NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE), 0, 0, INTER_NEAREST);
        cvtColor(resized, resized, CV_BGR2RGB);
        resized.convertTo(resized16f, CV_32F, 1/255.0);
        
        //get result from NCS
        if(!NCS.get_result(result))
        {
            NCS.print_error_code();
            break;
        }
            
        //get boxes and probs
        probs.clear();
        rects.clear();
        get_detection_boxes(result, resized.cols, resized.rows, 0.2, probs, rects);
        
        //non-maximum suppression
        do_nms(rects, probs, 1, 0.2);
        
        //Exit if any key pressed
        if (waitKey(1)!=-1)
        {
            break;
        }
    }
    
    //calculate fps
    double time = (getTickCount()-start)/getTickFrequency();
    cout<<"Frame rate: "<<nframes/time<<endl;
    
    cap.release();

    return 0;
}
