#include <inference_engine.hpp>

#include <opencv2/opencv.hpp>
#include <iostream>

#include "wrapper/vino_wrapper.hpp"

#include "./rpi_switch.h"
#if USE_RASPICAM
    #include <raspicam/raspicam.h>
#endif
#define BB_RAW_WIDTH                     1280
#define BB_RAW_HEIGHT                    960

using namespace std;
using namespace cv;
using namespace InferenceEngine;

/* function to parse SSD fetector output
 * @param predictions: output buffer of SSD net 
 * @param numPred: maximum number of SSD predictions (from net config)
 * @param w,h: target image height and width
 * @param thresh: detection threshold
 * @param probs, boxes: resulting confidences and bounding boxes
 */
void get_detection_boxes(const float* predictions, int numPred, int w, int h, float thresh, 
			 std::vector<float>& probs, std::vector<cv::Rect>& boxes)
{
    float score = 0;
    float cls = 0;
    float id = 0;
    
    //predictions holds numPred*7 values
    //data format: image_id, detection_class, detection_confidence, box_normed_x, box_normed_y, box_normed_w, box_normed_h
    for (int i=0; i<numPred; i++)
    {
      score = predictions[i*7+2];
      cls = predictions[i*7+1];
      id = predictions[i*7  ];
      if (id>=0 && score>thresh && cls<=1)
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
  NCSWrapper NCS(true);
  
  //Start communication with NCS
  if (!NCS.load_file("./models/face/vino"))
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

  

  //define raw frame and preprocessed frame
  Mat frame;
  Mat resized(NCS.netInputHeight, NCS.netInputWidth, CV_8UC3);
  resized = Scalar(0);
  
  float* result;
  
  //Capture-Render cycle
  int nframes=0;
  int64 start = getTickCount();
  
  vector<Rect> rects;
  vector<float> probs;
  for(;;)
  {
    nframes++;
    
    if (!NCS.load_tensor_nowait(resized))
      break;
    
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
    unsigned char* frame_data = Camera.getImageBufferData();
    frame = cv::Mat(BB_RAW_HEIGHT, BB_RAW_WIDTH, CV_8UC3, frame_data);
#else
    cap >> frame; 
#endif
    
    //transform next frame while NCS works
    if (frame.channels()==4)
      cvtColor(frame, frame, CV_BGRA2BGR);
    flip(frame, frame, 1);
    resize(frame, resized, Size(NCS.netInputWidth, NCS.netInputHeight));
    //cvtColor(resized, resized, CV_BGR2RGB);
    
    if (!NCS.get_result(result))
    {
      NCS.print_error_code();
      break;
    }
    
    //get boxes and probs
    probs.clear();
    rects.clear();
    get_detection_boxes(result, NCS.maxNumDetectedFaces, resized.cols, resized.rows, 0.5, probs, rects);
    
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