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
    
    // Allocate space for vectors A, B and C
    //------------------------------------------------
    size_t * size_A = new size_t[3]{1,2,2};
    
    // Get the truncated normal values in 1D array
    float * A = new float[size_A[1]*size_A[2]]{-0.723522,0.217484,-1.06868,0.806425};
    
//    m.truncated_normal(0, 1, size_A[1]*size_A[2], &A);
    for(u_sint batch = 0; batch < size_A[0]; batch++)
    {
        cout << "Matrix A:" << endl;
        u_sint s_i = batch*(size_A[1]*size_A[2]);
        for(u_sint i = s_i; i < (size_A[1]*size_A[2]); i+=size_A[2])
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
    
    
    // Fill the truncated normal values in the matrix mat1
    // Define the second matrix
    //-------------------------------------------------
    size_t * size_B = new size_t[2]{2,4};
    
    // Get the truncated normal values in 1D array
    cout << "Matrix B:" << endl;
    float * B = new float[size_B[0]*size_B[1]]{-1.3742,0.59425,0.891999,1.26432,-0.511385,0.93222,-0.56974,-1.240};
//    m.truncated_normal(0, 1, size_B[0]*size_B[1], &B);
    u_sint batch = 0;
    u_sint s_i = batch*(size_B[0]*size_B[1]);
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
    
    cout << "size_B: " << sizeof(size_B) << endl;
    
    // Copy the Buffer A and B to the device
    clStatus = clEnqueueWriteBuffer(opencl->command_queues[0], A_clmem,
                                    CL_TRUE, 0, sizeof(A),
                                    A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(opencl->command_queues[0], B_clmem,
                                    CL_TRUE, 0, sizeof(B),
                                    B, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(opencl->command_queues[0], A_clsize,
                                    CL_TRUE, 0, sizeof(size_A),
                                    size_A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(opencl->command_queues[0], B_clsize,
                                    CL_TRUE, 0, sizeof(size_B),
                                    size_B, 0, NULL, NULL);
    
   
    string sourceCode = readFile("/Users/rajkumar/Documents/OpenCL/Kernels/matmul.cl");
    
    // Create a program from the kernel source
    const char *source_chr = sourceCode.c_str();
    cl_program program = clCreateProgramWithSource(opencl->context,
                                                   1,&source_chr,
                                                   NULL,&clStatus);
    
    // Build the program
    clStatus = clBuildProgram(program, 1, opencl->device_list, NULL,
                              NULL, NULL);
    
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "matmul",
                                      &clStatus);
    
    size_t valueSize,cl_size_a,cl_size_b,cl_size_A,cl_size_B,cl_size_C;
    clGetMemObjectInfo(A_clsize, CL_MEM_SIZE, NULL, NULL, &valueSize);
    clGetMemObjectInfo(A_clsize, CL_MEM_SIZE, valueSize, &cl_size_a, NULL);
    clGetMemObjectInfo(B_clsize, CL_MEM_SIZE, NULL, NULL, &valueSize);
    clGetMemObjectInfo(B_clsize, CL_MEM_SIZE, valueSize, &cl_size_b, NULL);
    
    clGetMemObjectInfo(A_clmem, CL_MEM_SIZE, NULL, NULL, &valueSize);
    clGetMemObjectInfo(A_clmem, CL_MEM_SIZE, valueSize, &cl_size_A, NULL);
    clGetMemObjectInfo(B_clmem, CL_MEM_SIZE, NULL, NULL, &valueSize);
    clGetMemObjectInfo(B_clmem, CL_MEM_SIZE, valueSize, &cl_size_B, NULL);
    clGetMemObjectInfo(C_clmem, CL_MEM_SIZE, NULL, NULL, &valueSize);
    clGetMemObjectInfo(C_clmem, CL_MEM_SIZE, valueSize, &cl_size_C, NULL);
    
    // Set the arguments of the
    clStatus = clSetKernelArg(kernel, 0, cl_size_A,
                              (void *)&A_clmem);
    clStatus = clSetKernelArg(kernel, 1, cl_size_B,
                              (void *)&B_clmem);
    clStatus = clSetKernelArg(kernel, 2, cl_size_a,
                              (void *)&A_clsize);
    clStatus = clSetKernelArg(kernel, 3, cl_size_b,
                              (void *)&B_clsize);
    clStatus = clSetKernelArg(kernel, 4, cl_size_C,
                              (void *)&C_clmem);
    
    // Execute the OpenCL kernel on the list
    size_t global_size = size_C[0]*size_C[1]*size_C[2]; // Process the entire lists
    
    // Number of work items that make up a workgroup
    
    size_t * local_size = new size_t[1]{size_C[1]*size_C[2]};
    
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


