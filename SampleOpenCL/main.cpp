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
    size_t * size_A = new size_t[3]{3,2,2};
    
    // Get the truncated normal values in 1D array
    float * A = new float[size_A[0]*size_A[1]*size_A[2]];//{-0.723522,0.217484,-1.06868,0.806425};
    m.truncated_normal(0, 1, size_A[0]*size_A[1]*size_A[2], &A);
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
    
    
    // Fill the truncated normal values in the matrix mat1
    // Define the second matrix
    //-------------------------------------------------
    size_t * size_B = new size_t[2]{2,4};
    
    // Get the truncated normal values in 1D array
    cout << "Matrix B:" << endl;
    float * B = new float[size_B[0]*size_B[1]];//{-1.3742,0.59425,0.891999,1.26432,-0.511385,0.93222,-0.56974,-1.240};
    m.truncated_normal(0, 1, size_B[0]*size_B[1], &B);
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
    size_t A_clmem_size = sizeof(float)*(size_A[0]*size_A[1]*size_A[2]);
    cl_mem A_clmem = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY,
                                    A_clmem_size, NULL, &clStatus);
    size_t B_clmem_size = sizeof(float)*(size_B[0]*size_B[1]);
    cl_mem B_clmem = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY,
                                    B_clmem_size, NULL, &clStatus);
    size_t C_clmem_size = sizeof(float)*(size_C[0]*size_C[1]*size_C[2]);
    cl_mem C_clmem = clCreateBuffer(opencl->context, CL_MEM_WRITE_ONLY,
                                    C_clmem_size, NULL, &clStatus);
    size_t A_clsize_size = sizeof(size_t)*3;
    cl_mem A_clsize = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY,
                                     A_clsize_size, NULL, &clStatus);
    size_t B_clsize_size = sizeof(size_t)*2;
    cl_mem B_clsize = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY,
                                     B_clsize_size, NULL, &clStatus);
    
    
    // Copy the Buffer A and B to the device
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
    
   
    string sourceCode = readFile
                            ("/Users/rajkumarconjeevarammohan/Documents\ -\ "
                             "local/OpenCL/Kernels/matmul.cl");
    
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
    
    // Set the arguments of the
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
    
    // Execute the OpenCL kernel on the list
    cl_uint work_dim = 1;
    
    // Process the entire lists
    size_t *global_size = new size_t[work_dim]{size_C[0]*size_C[1]*size_C[2]};
//    size_t global_size = size_C[0]*size_C[1]*size_C[2];
    
    // Number of work items that make up a workgroup
    
//    size_t * local_size = new size_t[3]{size_C[1]*size_C[2],0,0};
//    size_t local_size = size_C[1]*size_C[2];
    
    clStatus = clEnqueueNDRangeKernel(opencl->command_queues[0],
                                      kernel,
                                      work_dim,
                                      NULL,
                                      global_size,
                                      NULL,
                                      0, NULL, NULL);
    
    // Read the cl memory C_clmem on device to the host variable C
    clStatus = clEnqueueReadBuffer(opencl->command_queues[0], C_clmem,
                                   CL_TRUE, 0, C_clmem_size, C, 0, NULL, NULL);
    
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


