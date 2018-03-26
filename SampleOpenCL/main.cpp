#include <stdio.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include "Matrix.h"
#include "OpenCLSetup.h"
#define VECTOR_SIZE 1024

using namespace std;

string readFile(string file_name)
{
    // Read the program source
    ifstream sourceFile(file_name);
    string sourceCode{ (istreambuf_iterator<char>(sourceFile)),
                       (istreambuf_iterator<char>()) };
    return sourceCode;
}

int main(void) {
    Matrix m;
    
    //------------------------------------------------
    // Define the first matrix A
    //------------------------------------------------
    u_sint * size_A = new u_sint[3]{1,2,2};
    float * A = new float[size_A[0]*size_A[1]*size_A[2]];
    
    // Fill the truncated normal values in the second matrix A
    m.truncated_normal(0, 1, size_A[0]*size_A[1]*size_A[2], &A);
    
    // Print the contents of A for debugging...
    cout << "Matrix A: " << endl;
    for(u_sint batch = 0; batch < size_A[0]; batch++)
    {
        printf("batch %d:\n",batch);
        u_sint s_i = batch*(size_A[1]*size_A[2]);
        printf("s_i: %d\n",s_i);
        for(u_sint i = s_i; i < s_i+(size_A[1]*size_A[2]); i+=size_A[2])
        {
            for(u_sint j = i; j < (i+size_A[2]); j++)
            {
                cout << A[j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    //-------------------------------------------------
    // Define the second matrix B
    //-------------------------------------------------
    u_sint * size_B = new u_sint[2]{2,4};
    float * B = new float[size_B[0]*size_B[1]];
    
    // Fill the truncated normal values in the second matrix B
    m.truncated_normal(0, 1, size_B[0]*size_B[1], &B);
    u_sint batch = 0;
    u_sint s_i = batch*(size_B[0]*size_B[1]);
    
    // Print the contents of B for debugging...
    cout << "Matrix B:" << endl;
    for(u_sint i = s_i; i < (size_B[0]*size_B[1]); i+=size_B[1])
    {
        for(u_sint j = i; j < (i+size_B[1]); j++)
        {
            cout << B[j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    
    //-------------------------------------------------
    // Define the third matrix to read back
    // the results
    //-------------------------------------------------
    u_sint * size_C = new u_sint[3]{size_A[0],size_A[1],size_B[1]};
    float * C = new float[size_C[0]*size_C[1]*size_C[2]];
    
    
    OpenCLSetup * opencl = new OpenCLSetup(1);
    cl_int clStatus;
    
    //-----------------------------------------------
    // Allocate space for variables on device memory
    //-----------------------------------------------
    size_t A_clmem_size = sizeof(float)*(size_A[0]*size_A[1]*size_A[2]);
    cl_mem A_clmem = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY,
                                    A_clmem_size, NULL, &clStatus);
    size_t B_clmem_size = sizeof(float)*(size_B[0]*size_B[1]);
    cl_mem B_clmem = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY,
                                    B_clmem_size, NULL, &clStatus);
    size_t C_clmem_size = sizeof(float)*(size_C[0]*size_C[1]*size_C[2]);
    cl_mem C_clmem = clCreateBuffer(opencl->context, CL_MEM_WRITE_ONLY,
                                    C_clmem_size, NULL, &clStatus);
    size_t A_clsize_size = sizeof(u_sint)*3;
    cl_mem A_clsize = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY,
                                     A_clsize_size, NULL, &clStatus);
    size_t B_clsize_size = sizeof(u_sint)*2;
    cl_mem B_clsize = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY,
                                     B_clsize_size, NULL, &clStatus);
    
    
    //--------------------------------------------------------
    // Copy the variables A,B,size_A,size_B vectors to cl_mem
    // buffers on the device
    //--------------------------------------------------------
    clStatus = clEnqueueWriteBuffer(opencl->command_queues[0], A_clmem,
                                    CL_TRUE, 0, A_clmem_size,
                                    A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(opencl->command_queues[0], B_clmem,
                                    CL_TRUE, 0, B_clmem_size,
                                    B, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(opencl->command_queues[0], A_clsize,
                                    CL_TRUE, 0, A_clsize_size,
                                    size_A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(opencl->command_queues[0], B_clsize,
                                    CL_TRUE, 0, B_clsize_size,
                                    size_B, 0, NULL, NULL);
   
    //--------------------------------------------------------
    // Read the source code
    //--------------------------------------------------------
    string sourceCode = readFile
                            ("/Users/rajkumar/Documents/OpenCL/Kernels/matmul.cl");
    
    //--------------------------------------------------------
    // Create a program from the kernel source
    //--------------------------------------------------------
    const char *source_chr = sourceCode.c_str();
    cl_program program = clCreateProgramWithSource(opencl->context,
                                                   1,&source_chr,
                                                   NULL,&clStatus);
    
    //--------------------------------------------------------
    // Build the program
    clStatus = clBuildProgram(program, 1, opencl->device_list, NULL,
                              NULL, NULL);
    //--------------------------------------------------------
    
    //--------------------------------------------------------
    // Create the OpenCL kernel
    //--------------------------------------------------------
    cl_kernel kernel = clCreateKernel(program, "matmul",
                                      &clStatus);
    
    //--------------------------------------------------------
    // Set the arguments for the kernel
    //--------------------------------------------------------
    clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem),
                              (void *)&A_clmem);
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem),
                              (void *)&B_clmem);
    clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem),
                              (void *)&A_clsize);
    clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem),
                              (void *)&B_clsize);
    clStatus = clSetKernelArg(kernel, 4, sizeof(cl_mem),
                              (void *)&C_clmem);
    
    //--------------------------------------------------------
    // Dimension of index space to execute the kernel
    //--------------------------------------------------------
    cl_uint work_dim = 1;
    
    //--------------------------------------------------------
    // Define the global and local size
    // Global size refers to total number of work items from
    // - all work groups.
    // Local size refers to the number of work items that make
    // up each work group.
    //--------------------------------------------------------
    size_t * global_size = new size_t[work_dim]{(size_t)size_C[0]*size_C[1]*size_C[2]};
    size_t * local_size = new size_t[work_dim]{(size_t)size_C[1]*size_C[2]};

    clStatus = clEnqueueNDRangeKernel(opencl->command_queues[0],
                                      kernel,
                                      work_dim,
                                      NULL,
                                      global_size,
                                      // Could keep local_size NULL for automatically selecting a value
                                      local_size,
                                      0, NULL, NULL);
    
    //--------------------------------------------------------
    // Read the contents of C_clmem buffer (i.e. the processed array)
    // from device to host memory.
    //--------------------------------------------------------
    clStatus = clEnqueueReadBuffer(opencl->command_queues[0], C_clmem,
                                   CL_TRUE, 0, C_clmem_size, C, 0, NULL, NULL);
    
    //--------------------------------------------------------
    // Print the received matrix C in matrix format for
    // debugging purposes.
    //--------------------------------------------------------
//     Print the matrix C
//    for(u_sint batch = 0; batch < size_A[0]; batch++)
//    {
//        cout << "For batch " << batch << ":" << endl;
//        u_sint s_i = batch*(size_C[1]*size_C[2]);
//        for(u_sint i = s_i; i < (size_C[1]*size_C[2]); i+=size_C[2])
//        {
//            for(u_sint j = i; j < (i+size_C[2]); j++)
//            {
//                cout << C[j] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
    for(u_sint i = 0; i < 8; i++)
        cout << C[i] << " ";
    cout << endl;
    
    //---------------------------------------------------------------
    // Finally release all OpenCL allocated objects and host buffers.
    // Clean up and wait for all the comands to complete.
    //---------------------------------------------------------------
    clStatus = clFlush(opencl->command_queues[0]);
    clStatus = clFinish(opencl->command_queues[0]);
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(A_clmem);
    clStatus = clReleaseMemObject(B_clmem);
    clStatus = clReleaseMemObject(C_clmem);
    
    free(A);
    free(B);
    free(C);
    free(opencl);
    return 0;
}


