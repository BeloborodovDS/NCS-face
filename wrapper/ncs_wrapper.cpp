#include "ncs_wrapper.hpp"
#include "fp16.h"

#include <iostream>
#include <fstream>

using namespace std;

//read whole graph file into buffer, return pointer to buffer, set filesize
void* readGraph(const char* filename, unsigned int* filesize)
{
    ifstream file;
    try
    {
      file.open(filename, ios::binary);
    }
    catch(...)
    {
      return NULL;
    }
    
    file.seekg (0, file.end);
    *filesize = file.tellg();
    file.seekg (0, file.beg);

    char* buffer = new char[*filesize];
    if (file.read(buffer, *filesize))
    {
	file.close();
	return (void*)buffer;
    }
    
    file.close();
    return NULL;
}

NCSWrapper::NCSWrapper(unsigned int input_num, unsigned int output_num, bool is_verbose)
{
    n_input = input_num;
    n_output = output_num;
    verbose = is_verbose;
    
    ncsCode = MVNC_OK;
    ncsDevice = NULL;
    ncsName = new char[100];
    graphSize = 0;
    graphData = NULL;
    ncsGraph = NULL;
    resultSize = 0;
    result16f = NULL;
    otherParam = NULL;
    input16f = new unsigned short[n_input];
    nres = 0;
    result = new float[n_output];
    
    is_init = false;
    is_allocate = false;
}

NCSWrapper::~NCSWrapper()
{
    if(ncsName)
      delete [] ncsName;
    ncsName = NULL;
    if(input16f)
      delete [] (unsigned short*)input16f;
    input16f = NULL;
    if (result)
      delete [] result;
    result = NULL;
    
    if (is_allocate)
    {
	ncsCode = mvncDeallocateGraph(ncsGraph);
	ncsGraph = NULL;
	if (ncsCode != MVNC_OK)
	{
	    if (verbose)
	      cout<<"Cannot deallocate graph: "<<ncsCode<<endl;
	}
    }
    
    if (is_init)
    {
	ncsCode = mvncCloseDevice(ncsDevice);
	ncsDevice = NULL;
	if (ncsCode != MVNC_OK)
	{
	    if (verbose)
	      cout<<"Cannot close NCS device, status: "<<ncsCode<<endl;
	}
    }    
    
    if(graphData)
      delete [] (char*)graphData;
    graphData = NULL;
    
}

bool NCSWrapper::load_file(const char* filename)
{
    //Get NCS name
    ncsCode = mvncGetDeviceName(0, ncsName, 100);
    if (ncsCode != MVNC_OK)
    {
        if (verbose)
	  cout<<"Cannot find NCS device, status: "<<ncsCode<<endl;
        return false;
    }
    if (verbose)
      cout<<"Found device named "<<ncsName<<endl;
    
    //Open NCS device via its name
    ncsCode = mvncOpenDevice(ncsName, &ncsDevice);
    if (ncsCode != MVNC_OK)
    {  
        if (verbose)
	  cout<<"Cannot open NCS device, status: "<<ncsCode<<endl;
        return false;
    }
    if (verbose)
      cout<<"Successfully opened device\n";
    is_init = true;
    
    //Get graph file size and data
    graphData = readGraph(filename, &graphSize);
    if (graphData==NULL)
    {
	if (verbose)
	  cout<<"Cannot open graph file\n";
	return false;
    }
    if (verbose)
      cout<<"Successfully loaded graph file, size is: "<<graphSize<<endl;
    
    //Allocate computational graph
    ncsCode = mvncAllocateGraph(ncsDevice, &ncsGraph, graphData, graphSize);
    if (ncsCode != MVNC_OK)
    {
        if (verbose)
	  cout<<"Cannot allocate graph, status: "<<ncsCode<<endl;
	return false;
    }
    if (verbose)
      cout<<"Successfully allocated graph\n";
    is_allocate = true;
    
    return true;
}


bool NCSWrapper::load_tensor(float* data, float*& output)
{
    //transform to 16f
    floattofp16((unsigned char*)input16f, data, n_input);
    
    //load image to NCS
    ncsCode = mvncLoadTensor(ncsGraph, input16f, n_input*sizeof(unsigned short), NULL);
    if (ncsCode != MVNC_OK)
    {
	if (verbose)
	  cout<<"Cannot load image to NCS, status: "<<ncsCode<<endl;
	output = NULL;
	return false;
    }
    
    //get result from NCS
    ncsCode = mvncGetResult(ncsGraph, &result16f, &resultSize, &otherParam);
    if (ncsCode != MVNC_OK)
    {
	if (verbose)
	  cout<<"Cannot retrieve result from NCS, status: "<<ncsCode<<endl;
	output = NULL;
	return false;
    }
    
    //Check result size
    nres = resultSize/sizeof(unsigned short);
    if (nres!=n_output)
    {
	if (verbose)
	  cout<<"Output shape mismatch! Expected/Real: "<<n_output<<"/"<<nres<<endl;
	output = NULL;
	return false;
    }
    
    //decode result
    fp16tofloat(result, (unsigned char*)result16f, nres);
    
    output = result;
    return true;
}

bool NCSWrapper::load_tensor_nowait(float* data)
{
    //transform to 16f
    floattofp16((unsigned char*)input16f, data, n_input);
    
    //load image to NCS
    ncsCode = mvncLoadTensor(ncsGraph, input16f, n_input*sizeof(unsigned short), NULL);
    if (ncsCode != MVNC_OK)
    {
	if (verbose)
	  cout<<"Cannot load image to NCS, status: "<<ncsCode<<endl;
	return false;
    }
    return true;
}

bool NCSWrapper::get_result(float*& output)
{
    //get result from NCS
    ncsCode = mvncGetResult(ncsGraph, &result16f, &resultSize, &otherParam);
    if (ncsCode != MVNC_OK)
    {
	if (verbose)
	  cout<<"Cannot retrieve result from NCS, status: "<<ncsCode<<endl;
	output = NULL;
	return false;
    }
    
    //Check result size
    nres = resultSize/sizeof(unsigned short);
    if (nres!=n_output)
    {
	if (verbose)
	  cout<<"Output shape mismatch! Expected/Real: "<<n_output<<"/"<<nres<<endl;
	output = NULL;
	return false;
    }
    
    //decode result
    fp16tofloat(result, (unsigned char*)result16f, nres);
    
    output = result;
    return true;
}



