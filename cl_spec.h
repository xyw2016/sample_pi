#ifndef __CL_SPEC__
#define __CL_SPEC__

#define __CL_ENABLE_EXCEPTIONS

#define USE_DEVICE_GPU

#include <cl.hpp>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <chrono>
#include <random>



#define USE_SINGLE_PRECISION

#ifdef USE_SINGLE_PRECISION
 typedef cl_float cl_real;   /*!< typedef cl_float to cl_real for easier switch from double to float */
 typedef cl_float2 cl_real2;   /*!< typedef cl_float to cl_real for easier switch from double to float */
 typedef cl_float4 cl_real4;
 typedef cl_float3 cl_real3;
 typedef cl_float8 cl_real8;

#else
 typedef cl_double cl_real;
 typedef cl_double2 cl_real2;
 typedef cl_double4 cl_real4;
 typedef cl_double3 cl_real3;
 typedef cl_double8 cl_real8;
#endif



class Spec
{

    public:

    Spec(int gpu_id, int N_THREADS, int iter);

    ~Spec();

    void initializeCL();

    cl::Context CreateContext(const cl_int& device_type);
    //cl::Context CreateContext();
    void AddProgram(const char * fname);

    void BuildPrograms(int program_id, const char* compile_options);  


    private:
    int gpu_id_;
    int N_THREADS_;
    int iter_;
    cl::Context context;
    std::vector<cl::Device> devices;
    std::vector<cl::Program> programs;
    cl::CommandQueue queue;


};



#endif