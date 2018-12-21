
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include "device_summary.h"
#include "convolution.h"
#include "param.h"
using namespace cl; 
typedef __SIZE_TYPE__  size_t;

 
	   std::vector<Platform> platforms;
           std::vector<Device> Devices; 
	  

int main() {

	std::size_t localWorkSize[2], globalWorkSize[2];
	

// constants
    const unsigned int imageW = 3072;
    const unsigned int imageH = 3072; 

///    
    cl_mem a_in_mem , b_in_mem , c_out_mem , d_out_mem;   //OpenCL memory buffer objects  
//**  
     ////*** Host_memory  
     cl_float*  a_in =   (cl_float *) malloc(imageW * imageH * sizeof(cl_float));
     cl_float*  b_in=    (cl_float *) malloc(FULL_FILTER   * sizeof(cl_float)); 
     cl_float*  d_out =  (cl_float *) malloc(imageW * imageH * sizeof(cl_float));
     cl_float*  c_out_host =  (cl_float *) malloc(imageW * imageH * sizeof(cl_float));
     cl_float*  d_out_host =  (cl_float *) malloc(imageW * imageH * sizeof(cl_float));

	//// Initializing host memory
	
       for(unsigned int i = 0; i < imageW * imageH; i++)
            a_in[i] = (cl_float)(rand() % 16);


      for(unsigned int i = 0; i < FULL_FILTER; i++)
            b_in[i] = (cl_float)(rand() % 16);

 
   try { 

   
	   // Get Plafrom and Device 
  
	  
	   /////
           device_query();      
	   /////
            
     //// Creating Context

              Context context(CL_DEVICE_TYPE_GPU, NULL);

      // Create a command queue and use the first device

        CommandQueue queue = CommandQueue(context, Devices[0]);
 
        // Read source file
        std::ifstream sourceFile("ConvolutionSeparable.cl");
        std::string sourceCode(
            std::istreambuf_iterator<char>(sourceFile),
            (std::istreambuf_iterator<char>()));
        Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
 
        // Make program of the source code in the context
        Program program = Program(context, source);
 
        // Build program for these specific devices
        program.build(Devices);
 
        // Make kernel
        Kernel ConvolutionRows_kernel(program, "Horizontalconv");
        Kernel ConvolutionColumns_kernel(program, "Verticalconv");
      
        // Create memory buffers
     Buffer a_in_mem = Buffer(context, CL_MEM_READ_ONLY,  imageW * imageH * sizeof(cl_float));
     Buffer b_in_mem = Buffer(context, CL_MEM_READ_ONLY,  FULL_FILTER * sizeof(cl_float));
     Buffer c_out_mem = Buffer(context, CL_MEM_WRITE_ONLY,  imageW * imageH * sizeof(cl_float));
     Buffer d_out_mem = Buffer(context, CL_MEM_WRITE_ONLY, imageW * imageH * sizeof(cl_float));


        
  
        // Copy lists A and B to the memory buffers
     queue.enqueueWriteBuffer(a_in_mem, CL_TRUE, 0, imageW * imageH * sizeof(cl_float), a_in);
     queue.enqueueWriteBuffer(b_in_mem, CL_TRUE, 0, FULL_FILTER * sizeof(cl_float), b_in);
 
        // Set arguments to kernel
        ConvolutionRows_kernel.setArg(0, c_out_mem);
        ConvolutionRows_kernel.setArg(1, a_in_mem);
        ConvolutionRows_kernel.setArg(2, b_in_mem);
        ConvolutionRows_kernel.setArg(3, imageW);
        ConvolutionRows_kernel.setArg(4, imageH);
        ConvolutionRows_kernel.setArg(5, imageW);
	


 
        // Run the kernel on specific ND range
	//
	localWorkSize[0] = H_LOCAL_X;
        localWorkSize[1] = H_LOCAL_Y;
        globalWorkSize[0] = imageW / H_OUT_SIZE;
        globalWorkSize[1] = imageH;
        
	NDRange global(globalWorkSize[0],globalWorkSize[1]);
        NDRange local( localWorkSize[0], localWorkSize[1]);
        queue.enqueueNDRangeKernel(ConvolutionRows_kernel, NullRange , global, local);

        
       	ConvolutionColumns_kernel.setArg(0, d_out_mem);
        ConvolutionColumns_kernel.setArg(1, c_out_mem);
        ConvolutionColumns_kernel.setArg(2, b_in_mem);
        ConvolutionColumns_kernel.setArg(3, imageW);
        ConvolutionColumns_kernel.setArg(4, imageH);
        ConvolutionColumns_kernel.setArg(5, imageW);
		
	
        localWorkSize[0] = V_LOCAL_X;
        localWorkSize[1] = V_LOCAL_Y;
        globalWorkSize[0] = imageW;
        globalWorkSize[1] = imageH / V_OUT_SIZE;
 
        NDRange global_1(globalWorkSize[0],globalWorkSize[1]);
        NDRange local_1( localWorkSize[0], localWorkSize[1]);
        queue.enqueueNDRangeKernel(ConvolutionColumns_kernel, NullRange , global_1, local_1);



 
        // Read buffer C into a local list
    
        queue.enqueueReadBuffer(d_out_mem, CL_TRUE, 0, imageW * imageH * sizeof(cl_float), d_out);


         // #Region 8: Result Validation
       Horizontalconv_CPU(c_out_host, a_in , b_in, imageW, imageH, HALF_FILTER );
       Verticalconv_CPU(d_out_host, c_out_host , b_in, imageW, imageH, HALF_FILTER);
      
/*      
*/
  

	FILE *fp;
	fp = fopen("output.txt","w"); 
       int  err=0;
	for(unsigned int i = 0 ; i < imageW * imageH ; i++)
          
		
	{        
             if(d_out[i]-d_out_host[i] !=0 )
		
		{

      	 	err=1;

	    fprintf(fp,"ERROR: Host output :%f and Device output:%f and Pixel:%f \n", d_out_host[i], d_out[i], i);
        }
  }

	if(err==0)
	
	{
	fprintf(fp,"\n\n Test is Passed, performance metrics can be analyzed using SCOREP trace files (make with scorep=1). \n\n");	

	}


        printf("\n Please open output.txt file for more information\n");

       
       fclose(fp);

   }

       catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }
    
	return 0;

   
}
