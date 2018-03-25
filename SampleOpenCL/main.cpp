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
    cout << "File Name: " << file_name << endl;
    ifstream sourceFile(file_name);
    
    
    string sourceCode{ (istreambuf_iterator<char>(sourceFile)),
                       (istreambuf_iterator<char>()) };
    cout << "Kernel " << sourceCode << "\n" << endl;
    return sourceCode;
}

int main(void) {
    Matrix m;
    
    // Allocate space for vectors A, B and C
    //------------------------------------------------
    size_t * size_A = new size_t[3]{1,2,2};
    
    // Get the truncated normal values in 1D array
    float * A = new float[size_A[1]*size_A[2]];
    m.truncated_normal(0, 1, size_A[1]*size_A[2], &A);
    //-------------------------------------------------
    
    
    // Fill the truncated normal values in the matrix mat1
    // Define the second matrix
    //-------------------------------------------------
    size_t * size_B = new size_t[2]{2,4};
    
    // Get the truncated normal values in 1D array
    float * B = new float[size_B[0]*size_B[1]];
    m.truncated_normal(0, 1, size_B[0]*size_B[1], &B);
    //-------------------------------------------------
 
    // Define the third matrix to read back
    // the results
    //-------------------------------------------------
    size_t * size_C = new size_t[3]{size_A[0],size_A[1],size_B[1]};
    float * C = new float[size_C[0]*size_C[1]*size_C[2]];
    //-------------------------------------------------
    
    OpenCLSetup * opencl = new OpenCLSetup(1);
    cl_int clStatus;
    
    // Create memory buffers on the device for each vector
    cl_mem A_clmem = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY,
                                    sizeof(A), NULL, &clStatus);
    cl_mem B_clmem = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY,
                                    sizeof(B), NULL, &clStatus);
    cl_mem C_clmem = clCreateBuffer(opencl->context, CL_MEM_WRITE_ONLY,
                                    sizeof(C), NULL, &clStatus);
    cl_mem A_clsize = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY,
                                     sizeof(size_A), NULL, &clStatus);
    cl_mem B_clsize = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY,
                                     sizeof(size_B), NULL, &clStatus);
    
    // Copy the Buffer A and B to the device
    clStatus = clEnqueueWriteBuffer(opencl->command_queues[0], A_clmem,
                                    CL_TRUE, 0, VECTOR_SIZE * sizeof(float),
                                    A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(opencl->command_queues[0], B_clmem,
                                    CL_TRUE, 0, VECTOR_SIZE * sizeof(float),
                                    B, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(opencl->command_queues[0], A_clsize,
                                    CL_TRUE, 0, sizeof(size_A),
                                    size_A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(opencl->command_queues[0], B_clsize,
                                    CL_TRUE, 0, sizeof(size_B),
                                    size_B, 0, NULL, NULL);
    
   
    string sourceCode = readFile("/Users/rajkumar/Documents/OpenCL/Kernels/matmul.cl");
    
    // Create a program from the kernel source
    size_t len = (size_t)sourceCode.length();
    size_t * len_ptr = &(len);
    cl_program program = clCreateProgramWithSource(opencl->context,
                                                   1,(const char**)&sourceCode,
                                                   NULL,&clStatus);
    
    // Build the program
    clStatus = clBuildProgram(program, 1, opencl->device_list, NULL,
                              NULL, NULL);
    
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "matmul",
                                      &clStatus);
    
    // Set the arguments of the
    clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem),
                              (void *)&A_clmem);
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem),
                              (void *)&B_clmem);
    clStatus = clSetKernelArg(kernel, 2, sizeof(cl_ushort),
                              (void *)&size_A);
    clStatus = clSetKernelArg(kernel, 3, sizeof(cl_ushort),
                              (void *)&size_B);
    clStatus = clSetKernelArg(kernel, 4, sizeof(cl_mem),
                              (void *)&C_clmem);
    
    // Execute the OpenCL kernel on the list
    size_t global_size = size_C[0]*size_C[1]*size_C[2]; // Process the entire lists
    
    // Number of work items that make up a workgroup
    
    size_t * local_size = new size_t[1]{100};
    
    clStatus = clEnqueueNDRangeKernel(opencl->command_queues[0],
                                      kernel,
                                      1,NULL,
                                      &global_size,
                                      local_size,
                                      0, NULL, NULL);
    
    // Read the cl memory C_clmem on device to the host variable C
    clStatus = clEnqueueReadBuffer(opencl->command_queues[0], C_clmem,
                                   CL_TRUE, 0, sizeof(C), C, 0, NULL, NULL);
    
    // Clean up and wait for all the comands to complete.
    clStatus = clFlush(opencl->command_queues[0]);
    clStatus = clFinish(opencl->command_queues[0]);
    
    // Print the matrix C
    for(u_sint batch = 0; batch < size_A[0]; batch++)
    {
        cout << "For batch " << batch << ":" << endl;
        u_sint s_i = batch*(size_C[1]*size_C[2]);
        for(u_sint i = s_i; i < (size_C[1]*size_C[2]); i+=size_C[2])
        {
            for(u_sint j = i; j < (i+size_C[2]); j++)
            {
                cout << C[j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    
    // Finally release all OpenCL allocated objects and host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(A_clmem);
    clStatus = clReleaseMemObject(B_clmem);
    clStatus = clReleaseMemObject(C_clmem);
    
    free(A);
    free(B);
    free(C);
    free(opencl);
    getchar();
    return 0;
}


