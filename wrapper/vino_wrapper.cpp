#include "vino_wrapper.hpp"

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include <inference_engine.hpp>

using namespace std;
using namespace InferenceEngine;
using namespace cv;

NCSWrapper::NCSWrapper(bool is_verbose)
{
  verbose = is_verbose;
  
  inputName = "";
  outputName = "";
  request = nullptr;
  inputBlob = nullptr;
  netInputWidth = -1;
  netInputHeight = -1;
  netInputChannels = -1;
  ncsCode = StatusCode::OK;
}

bool NCSWrapper::load_file(string filename)
{
  //get plugin (i.e dynamic library) for NCS
  //Empty path means to search in LD_LIBRARY_PATH
  InferencePlugin plugin = PluginDispatcher({""}).getPluginByDevice("MYRIAD");
  
  if (verbose)
  {
    //print plugin version just for fun
    const InferenceEngine::Version *pluginVersion = nullptr;
    pluginVersion = plugin.GetVersion();
    cout << "MYRIAD plugin version: " << pluginVersion->description <<" "<< pluginVersion->buildNumber 
	    <<" "<<(pluginVersion->apiVersion).major <<"."<<(pluginVersion->apiVersion).minor << endl;
  }
  
  //object responsible for reading and configuring net
  CNNNetReader netReader;
  try
  {
    netReader.ReadNetwork(filename+".xml"); //network topology
    netReader.ReadWeights(filename+".bin"); //network weights
    netReader.getNetwork().setBatchSize(1); //for NCS batches are always of size 1
  }
  catch (...)
  {
    if (verbose)
      cout<<"Cannot open network files: "<<filename+".xml"<< " or "<<filename+".bin"<<endl;
    return false;
  }
  
  //we can set input type to unsigned char: conversion will be performed on device
  netReader.getNetwork().getInputsInfo().begin()->second->setPrecision(Precision::U8);
  //get input and output names and their info structures
  inputName = netReader.getNetwork().getInputsInfo().begin()->first;
  outputName = netReader.getNetwork().getOutputsInfo().begin()->first;
  OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
  InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
  DataPtr &outputData = (outputInfo.begin()->second);
  
  //get output shape: (batch(1) x 1 x maxNumDetectedFaces x faceDescriptionLength(7))
  const SizeVector outputDims = outputData->getTensorDesc().getDims();
  maxNumDetectedFaces = outputDims[2];
  //set input type to float32: calculations are all in float16, conversion is performed on device
  outputData->setPrecision(Precision::FP32);
  
  try //compile net for NCS and load into the device
  {
    net = plugin.LoadNetwork(netReader.getNetwork(), {});
  }
  catch (...)
  {
    if (verbose)
      cout << "Cannot load network into NCS, probably device not connected\n";
    return false;
  }
  
  try
  {
    //perform single inference to get input shape (a hack)
    request = net.CreateInferRequestPtr(); //open inference request
    //we need the blob size: (batch(1) x channels(3) x H x W)
    inputBlob = request->GetBlob(inputName);
    SizeVector blobSize = inputBlob->getTensorDesc().getDims();
    netInputWidth = blobSize[3];
    netInputHeight = blobSize[2];
    netInputChannels = blobSize[1];
    if (verbose)
      cout<<"Network dims (H x W x C): "<<netInputHeight<<" x "<<netInputWidth<<" x "<<netInputChannels<<endl;
    request->Infer(); //close request
  }
  catch (...)
  {
    if (verbose)
      cout<<"Failed first inference!\n";
    return false;
  }
    
  return true;
}


bool NCSWrapper::load_tensor(Mat &data, float*& output)
{    
  //create request, get data blob
  request = net.CreateInferRequestPtr();
  inputBlob = request->GetBlob(inputName);
  unsigned char* blobData = inputBlob->buffer().as<unsigned char*>();
  
  //copy from resized frame to network input
  for (int c = 0; c < netInputChannels; c++)
    for (int h = 0; h < netInputHeight; h++)
      for (int w = 0; w < netInputWidth; w++)
	  blobData[c * netInputWidth * netInputHeight + h * netInputWidth + w] = data.at<cv::Vec3b>(h, w)[c];
  
  //start synchronous inference
  request->Infer();
  output = request->GetBlob(outputName)->buffer().as<float*>();
  
  return true;
}

bool NCSWrapper::load_tensor_nowait(Mat &data)
{
  //create request, get data blob
  request = net.CreateInferRequestPtr();
  inputBlob = request->GetBlob(inputName);
  unsigned char* blobData = inputBlob->buffer().as<unsigned char*>();
  
  //copy from resized frame to network input
  int wh = netInputHeight*netInputWidth;
  for (int c = 0; c < netInputChannels; c++)
    for (int h = 0; h < wh; h++)
	  blobData[c * wh + h] = data.data[netInputChannels*h + c];
  
  //start asynchronous inference
  request->StartAsync();
  
  return true;
}

bool NCSWrapper::get_result(float*& output)
{
  //wait for results
  ncsCode = request->Wait(IInferRequest::WaitMode::RESULT_READY);
  output = request->GetBlob(outputName)->buffer().as<float*>();
  
  if (ncsCode != StatusCode::OK)
  {
    if (verbose)
      cout<<"get_result failed!\n";
    return false;
  }

  return true;    
}

void NCSWrapper::print_error_code()
{
    cout<<"NCSWrapper error report:\n";
    
    if (ncsCode == StatusCode::OK)
    {
        cout<<"Everything is fine, no error\n";
    }
    else if (ncsCode == StatusCode::GENERAL_ERROR)
    {
        cout<<"GENERAL_ERROR\n";
    }
    else if (ncsCode == StatusCode::NOT_IMPLEMENTED)
    {
        cout<<"NOT_IMPLEMENTED\n";
    }
    else if (ncsCode == StatusCode::NETWORK_NOT_LOADED)
    {
        cout<<"NETWORK_NOT_LOADED\n";
    }
    else if (ncsCode == StatusCode::PARAMETER_MISMATCH)
    {
        cout<<"PARAMETER_MISMATCH\n";
    }
    else if (ncsCode == StatusCode::NOT_FOUND)
    {
        cout<<"NOT_FOUND\n";
    }
    else if (ncsCode == StatusCode::OUT_OF_BOUNDS)
    {
        cout<<"OUT_OF_BOUNDS\n";
    }
    else if (ncsCode == StatusCode::UNEXPECTED)
    {
        cout<<"UNEXPECTED exception\n";
    }
    else if (ncsCode == StatusCode::REQUEST_BUSY)
    {
        cout<<"REQUEST_BUSY\n";
    }
    else if (ncsCode == StatusCode::RESULT_NOT_READY)
    {
        cout<<"RESULT_NOT_READY\n";
    }
    else if (ncsCode == StatusCode::NOT_ALLOCATED)
    {
        cout<<"NOT_ALLOCATED\n";
    }
    else if (ncsCode == StatusCode::INFER_NOT_STARTED)
    {
        cout<<"INFER_NOT_STARTED\n";
    }
    else if (ncsCode == StatusCode::NETWORK_NOT_READ)
    {
        cout<<"NETWORK_NOT_READ\n";
    }
    else
    {
        cout<<"Some other error occured, unknown code "<<ncsCode<<endl;
    }
}

