#ifndef NCS_WRAPPER_HEADER
#define NCS_WRAPPER_HEADER

#include <iostream>
#include <fstream>

#include <mvnc.h>

void* readGraph(const char* filename, unsigned int* filesize);

class NCSWrapper 
{
public:
    /* Construct wrapper
     * @param input_size: total network input 
     * @param output_size: total network output_size
     */
    NCSWrapper(unsigned int input_num, unsigned int output_num, bool is_verbose=true);
    
    /* Destructor: deallocate all resources 
     */
    ~NCSWrapper();
    
    /* find and open NCS, load graph file, allocate graph
     * @param filename: name of compiled graph file
     * @return: true if success, else false
     */ 
    bool load_file(const char* filename);
    
    /* load data into NCS, get result
     * @param data: pointer to input data
     * @param output: reference to pointer for output data
     * @return: true if success, else false
     */
    bool load_tensor(float* data, float*& output);
    
    /* load data into NCS without waiting for result
     * @param data: pointer to input data 
     * @return: true if success, else false
     */
    bool load_tensor_nowait(float* data);
    
    /* get result from NCS after calling load_tensor_nowait(...)
     * @param output: reference to pointer for output data
     * @return: true if success, else false
     */
    bool get_result(float*& output);
    
    //return code for MVNC functions
    mvncStatus ncsCode;
    //device handle
    void* ncsDevice;
    //device name
    char* ncsName;
    //graph file size
    unsigned int graphSize;
    //graph file buffer
    void* graphData;
    //graph handle
    void* ncsGraph;
    //result size in bytes
    unsigned int resultSize;
    //result buffer (float 16)
    void* result16f;
    //user parameters
    void* otherParam;
    //input buffer (float 16) 
    void* input16f;
    //number of obtained outputs
    unsigned int nres;
    //result buffer (float)
    float* result;
    
    //number of inputs and outputs
    unsigned int n_input, n_output;
    
    //if true, output text info to stdout
    bool verbose;
    
    //for destructor
    bool is_init;
    bool is_allocate;
    
};

#endif
