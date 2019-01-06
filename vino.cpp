#include <inference_engine.hpp>

#include <opencv2/opencv.hpp>
#include <iostream>

#include "./rpi_switch.h"
#if USE_RASPICAM
    #include <raspicam/raspicam.h>
#endif
#define BB_RAW_WIDTH                     1280
#define BB_RAW_HEIGHT                    960

using namespace std;
using namespace cv;
using namespace InferenceEngine;


void get_detection_boxes(const float* predictions, int numPred, int w, int h, float thresh, 
			 std::vector<float>& probs, std::vector<cv::Rect>& boxes)
{
    float score = 0;
    float cls = 0;
    float id = 0;
    
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
  //get plugin (i.e dynamic library) for NCS
  //Empty path means to search in LD_LIBRARY_PATH
  InferencePlugin plugin = PluginDispatcher({""}).getPluginByDevice("MYRIAD");
  
  //print plugin version just for fun
  const InferenceEngine::Version *pluginVersion = nullptr;
  pluginVersion = plugin.GetVersion();
  cout << "MYRIAD plugin version: " << pluginVersion->description <<" "<< pluginVersion->buildNumber 
	  <<" "<<(pluginVersion->apiVersion).major <<"."<<(pluginVersion->apiVersion).minor << endl;
	  
  //object responsible for reading and configuring net
  CNNNetReader netReader;
  string modelBasePath = "./models/face/vino";
  netReader.ReadNetwork(modelBasePath+".xml"); //network topology
  netReader.ReadWeights(modelBasePath+".bin"); //network weights
  netReader.getNetwork().setBatchSize(1);      //for NCS batches are always of size 1
  
  //we can set input type to unsigned char: conversion will be performed on device
  netReader.getNetwork().getInputsInfo().begin()->second->setPrecision(Precision::U8);
  //get input and output names and their info structures
  string inputName = netReader.getNetwork().getInputsInfo().begin()->first;
  string outputName = netReader.getNetwork().getOutputsInfo().begin()->first;
  OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
  InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
  DataPtr &outputData = (outputInfo.begin()->second);
  
  //get output shape: (batch(1) x 1 x maxNumDetectedFaces x faceDescriptionLength(7))
  const SizeVector outputDims = outputData->getTensorDesc().getDims();
  int maxNumDetectedFaces = outputDims[2];
  //set input type to float32: calculations are all in float16, conversion is performed on device
  outputData->setPrecision(Precision::FP32);
  
  std::map<std::string, std::string> config; //empty
  ExecutableNetwork net;
  try //compile net for NCS and load into the device
  {
    net = plugin.LoadNetwork(netReader.getNetwork(), config);
  }
  catch (...)
  {
    cout << "Cannot load network into NCS, probably device not connected\n";
    return 0;
  }
  
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

  //perform single inference to get input shape (a hack)
  InferRequest::Ptr request = nullptr;
  request = net.CreateInferRequestPtr(); //open inference request
  //we need the blob size: (batch(1) x channels(3) x H x W)
  Blob::Ptr inputBlob = request->GetBlob(inputName);
  SizeVector blobSize = inputBlob->getTensorDesc().getDims();
  int netInputWidth = blobSize[3];
  int netInputHeight = blobSize[2];
  int netInputChannels = blobSize[1];
  cout<<"Network dims (H x W x C): "<<netInputHeight<<" x "<<netInputWidth<<" x "<<netInputChannels<<endl;
  request->Infer(); //close request

  //define raw frame and preprocessed frame
  Mat frame;
  Mat resized(netInputHeight, netInputWidth, CV_8UC3);
  resized = Scalar(0);
  
  const float* result;
  
  //Capture-Render cycle
  int nframes=0;
  int64 start = getTickCount();
  
  vector<Rect> rects;
  vector<float> probs;
  for(;;)
  {
    nframes++;
	
    //create request, get data blob
    request = net.CreateInferRequestPtr();
    inputBlob = request->GetBlob(inputName);
    unsigned char* blobData = inputBlob->buffer().as<unsigned char*>();
    
    //copy from resized frame to network input
    for (int c = 0; c < netInputChannels; c++)
      for (int h = 0; h < netInputHeight; h++)
	for (int w = 0; w < netInputWidth; w++)
	    blobData[c * netInputWidth * netInputHeight + h * netInputWidth + w] = resized.at<cv::Vec3b>(h, w)[c];
    
    //start asynchronous inference
    request->StartAsync();
    
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
    resize(frame, resized, Size(netInputWidth, netInputHeight));
    //cvtColor(resized, resized, CV_BGR2RGB);
    
    //wait for results
    request->Wait(IInferRequest::WaitMode::RESULT_READY);
    result = request->GetBlob(outputName)->buffer().as<float*>();
    
    //get boxes and probs
    probs.clear();
    rects.clear();
    get_detection_boxes(result, maxNumDetectedFaces, resized.cols, resized.rows, 0.5, probs, rects);
    
    //Exit if any key pressed
    if (waitKey(1)!=-1)
    {
      break;
    }
  }
  
  //calculate fps
  double time = (getTickCount()-start)/getTickFrequency();
  cout<<"Frame rate: "<<nframes/time<<endl;

  return 0;
}