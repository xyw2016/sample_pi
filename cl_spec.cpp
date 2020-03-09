#include "cl_spec.h"

Spec::Spec(int gpu_id, int N_THREADS, int iter){
    gpu_id_ = gpu_id;
    N_THREADS_ = N_THREADS;
    iter_ = iter;
    initializeCL();
}

void Spec::initializeCL(){
   try {

       cl_int device_type= CL_DEVICE_TYPE_GPU;
#ifdef USE_DEVICE_GPU
        device_type = CL_DEVICE_TYPE_GPU;
#endif
       std::cout<< CL_DEVICE_TYPE_CPU << "  "<< CL_DEVICE_TYPE_GPU<<" "<<device_type<<std::endl;
       context = CreateContext(device_type);

       devices = context.getInfo<CL_CONTEXT_DEVICES>();

       for (std::vector<cl::Device>::size_type i=0; i!=devices.size(); i++ ){
           std::cout <<"#" <<devices[i].getInfo<CL_DEVICE_NAME>()<<std::endl;
           std::cout<<"#Max compute units = "<< devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()<<std::endl;
       }

       std::stringstream compile_options;
       std::string dev_vendor = devices[gpu_id_].getInfo<CL_DEVICE_VENDOR>();
       std::cout<<"#using device = "<<dev_vendor<<std::endl;

       int LenVector = devices[gpu_id_].getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>();
       std::cout <<"#preferred vector width float = "<< LenVector<<std::endl; 
     

       if(sizeof(cl_real) == 4){
           compile_options << "-D USE_SINGGLE_PRECISION"<<" ";
       }
       // for hydro, need more options

       queue = cl::CommandQueue( context, devices[gpu_id_], CL_QUEUE_PROFILING_ENABLE );

       AddProgram("sample_pi.cl");
       BuildPrograms(0,compile_options.str().c_str() );

       cl::Kernel kernel_samplepi = cl::Kernel(programs.at(0), "sample_pi");
       
       std::vector<cl_ulong> seed(N_THREADS_);
	   long long rnd_init = std::chrono::system_clock::now().time_since_epoch().count();

       cl::Buffer cl_seed(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N_THREADS_*sizeof(cl_ulong), seed.data());
	   cl::Buffer cl_res(context, CL_MEM_WRITE_ONLY, N_THREADS_* sizeof(cl_uint));

       kernel_samplepi.setArg(0,iter_);
       kernel_samplepi.setArg(1,cl_seed);
       kernel_samplepi.setArg(2,cl_res);

       cl::Event event;
       cl::NDRange globalSize = cl::NDRange( N_THREADS_ ); 
       cl::NDRange localSize = cl::NDRange( 1 );
       queue.enqueueNDRangeKernel(kernel_samplepi, cl::NullRange, \
                            globalSize, localSize, NULL, &event); //256*64, 256
       event.wait();

       std::vector<cl_uint> res(N_THREADS_);
       queue.enqueueReadBuffer( cl_res, CL_TRUE, 0, N_THREADS_* sizeof(cl_uint),res.data() );
       
       cl_real sum = 0.0;
       for ( int i = 0; i < N_THREADS_ ; i++){
           sum+= res[i];
       }

       cl_real pi = 3.14159265359;
       sum = sum/(N_THREADS_*iter_)*4.0;

       std::cout<<sum<<std::endl;
       std::cout<<"Difference: "<<abs(sum-pi)<<std::endl;

       


       
       

    }
    catch (cl::Error & err){
        std::cerr<<"Error: "<< err.what()<<"("<<err.err()<<")\n";
    }
}



cl::Context Spec::CreateContext(const cl_int& device_type)
{

    std::vector<cl::Platform> platforms;

    cl::Platform::get(&platforms);
    if(platforms.size()==0){
        std::cerr<<"NO PLATFORM FOUND!!! \n";
        exit(-1);
    }
    else{
        for(int i = 0 ;i < platforms.size(); i ++){
            std::vector<cl::Device> supportDevices;
            platforms.at(i).getDevices(CL_DEVICE_TYPE_ALL, &supportDevices);
            for (int j = 0; j< supportDevices.size();j++){
                if( supportDevices.at(j).getInfo<CL_DEVICE_TYPE>() == device_type )
                {
                    std::cout<<"#Found device "<<supportDevices[j].getInfo<CL_DEVICE_NAME>()<<" on platform "<< i <<std::endl;
                    cl_context_properties properties[] = 
                    { CL_CONTEXT_PLATFORM, (cl_context_properties) (platforms.at(i))(), 0 };
                    return cl::Context(device_type, properties);
                }
            }
        }
    }
    std::cerr<<"No platform support device type "<< device_type<<std::endl;
    exit(-1);


}

void Spec::AddProgram(const char * fname){
    std::ifstream kernelFile( fname );
    if( !kernelFile.is_open() ) std::cerr<<"Open "<<fname << " failed!"<<std::endl;

    std::string sprog( std::istreambuf_iterator<char> (kernelFile), (std::istreambuf_iterator<char> ()) );
    cl::Program::Sources prog(1, std::make_pair(sprog.c_str(), sprog.length()));
    programs.push_back( cl::Program( context, prog ) );


}

void Spec::BuildPrograms( int i, const char * compile_options )
{ //// build programs and output the compile error if there is
    //for(std::vector<cl::Program>::size_type i=0; i!=programs.size(); i++)
    {
        try{
            programs.at(i).build(devices, compile_options);
        }
        catch(cl::Error & err){
            std::cerr<<err.what()<<"("<<err.err()<<")\n"<< programs.at(i).getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[gpu_id_]);
        }
    }

}





Spec::~Spec(){

}