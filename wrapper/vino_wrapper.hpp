#ifndef VINO_WRAPPER_HEADER
#define VINO_WRAPPER_HEADER

#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

#include <inference_engine.hpp>

using namespace InferenceEngine;
using namespace cv;
using namespace std;

class NCSWrapper 
{
public:
  /* Construct wrapper
    * @param is_verbose: if true, prints debug messages
    */
  NCSWrapper(bool is_verbose=true);
  
  /* Destructor: deallocate all resources 
    */
  //~NCSWrapper();
  
  /* find and open NCS, allocate and load model
    * @param filename: name of openvino model without extension, assumed to be pair "filename.xml" and "filename.bin"
    * @return: true if success, else false
    */ 
  bool load_file(string filename);
  
  /* load data into NCS, get result
    * @param data: 8UC3 Mat of appropriate size
    * @param output: reference to pointer for output data
    * @return: true if success, else false
    */
  bool load_tensor(Mat &data, float*& output);
  
  /* load data into NCS without waiting for result
    * @param data: 8UC3 Mat of appropriate size 
    * @return: true if success, else false
    */
  bool load_tensor_nowait(Mat &data);
  
  /* get result from NCS after calling load_tensor_nowait(...)
    * @param output: reference to pointer for output data
    * @return: true if success, else false
    */
  bool get_result(float*& output);
  
  /* print OpenVINO error code after Wait or Infer
   */
  void print_error_code();
  
  //input, output names
  string inputName;
  string outputName;
  //network itself
  ExecutableNetwork net;
  //inference request
  InferRequest::Ptr request;
  //input data blob
  Blob::Ptr inputBlob;
  //input shape
  int netInputWidth;
  int netInputHeight;
  int netInputChannels;
  //output shape
  int maxNumDetectedFaces;
  
  StatusCode ncsCode;
  
  //if true, output text info to stdout
  bool verbose;
    
};

#endif
