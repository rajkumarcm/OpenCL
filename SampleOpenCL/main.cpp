#include <stdio.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#define VECTOR_SIZE 1024

//OpenCL kernel which is run for every work item created.
const char *saxpy_kernel =
"float mul(float alpha, float A) { \n"
"  return alpha*A;} \n"

"__kernel \n"
"void saxpy_kernel(float alpha, \n"
" __global float *A, \n"
" __global float *B, \n"
" __global float *C) \n"
"{ \n"
" //Get the index of the work-item \n"
" int index = get_global_id(0); \n"
" C[index] = mul(alpha,A[index]) + B[index]; \n"
"} \n";

void get_platforms(cl_platform_id** platforms, cl_uint& num_platforms)
{
    //// Get platform and device information
    //cl_uint num_platforms;
    
    //Set up the Platform
    cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    *platforms = (cl_platform_id *)
    malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, *platforms, NULL);
    
    printf("The following platforms were found:\n");
    for (unsigned short int i = 0; i < num_platforms; i++)
    {
        size_t value_size;
        char* value;
        cl_platform_id platform = *platforms[i];
        // Get platform name
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &value_size);
        value = (char *) malloc(sizeof(char)*value_size);
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, value_size, value, NULL);
        printf("Platform Name: %s \n", value);
        
        // Get platform version
        clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, NULL, &value_size);
        value = (char *) malloc(sizeof(char)*value_size);
        clGetPlatformInfo(platform, CL_PLATFORM_VERSION, value_size, value, NULL);
        printf("Platform Version: %s \n", value);
        
        // Get platform vendor
        clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, NULL, &value_size);
        value = (char *)malloc(sizeof(char)*value_size);
        clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, value_size, value, NULL);
        printf("Platform Vendor: %s \n\n", value);
    }
}

void get_devices(cl_platform_id* platforms, const cl_uint num_platforms,
                 cl_device_id** device_list, cl_uint& num_devices)
{
    //Get the devices list and choose the device you want to run on
    cl_int clStatus;
    
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0,
                              NULL, &num_devices);
    *device_list = (cl_device_id *)
    malloc(sizeof(cl_device_id)*num_devices);
    
    clStatus = clGetDeviceIDs(platforms[0],
                              CL_DEVICE_TYPE_ALL, num_devices, *device_list, NULL);
    
    printf("The following devices were found under the platform: \n");
    for (unsigned short int i = 0; i < num_devices; i++)
    {
        size_t value_size;
        char* value;
        cl_device_id device = (*device_list)[i];
        printf("Device %d\n",i);
        clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &value_size);
        value = (char *) malloc(sizeof(char)*value_size);
        clGetDeviceInfo(device, CL_DEVICE_NAME, value_size, value, NULL);
        printf("Name: %s \n", value);
        
        clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, NULL, &value_size);
        value = (char *)malloc(sizeof(char)*value_size);
        clGetDeviceInfo(device, CL_DEVICE_VENDOR, value_size, value, NULL);
        printf("Vendor: %s \n", value);
        
        cl_ulong* global_size;
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, 0, NULL, &value_size);
        global_size = (cl_ulong *)malloc(sizeof(cl_ulong)*value_size);
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, value_size, global_size, NULL);
        
        double temp_size = ((*global_size / 1024) / 1024) / 1024;
        printf("Global Memory: %.2fGB \n",temp_size);
        
        cl_uint* clock_freq;
        clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, 0, NULL, &value_size);
        clock_freq = (cl_uint *)malloc(sizeof(cl_uint)*value_size);
        clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, value_size, clock_freq, NULL);
        printf("Max Clock Frequency: %dMHz \n", *clock_freq);
        
        cl_uint* compute_units;
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, 0, NULL, &value_size);
        compute_units = (cl_uint*)malloc(sizeof(cl_uint)*value_size);
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, value_size, compute_units, NULL);
        printf("Max Compute Units: %d \n", *compute_units);
        
        size_t* max_work_group_size;
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, NULL, &value_size);
        max_work_group_size = (size_t *) malloc(sizeof(size_t)*value_size);
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, value_size, max_work_group_size, NULL);
        printf("Max Work Group Size: %d \n", (int)(*max_work_group_size));
        
        size_t* max_work_item_sizes;
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &value_size);
        max_work_item_sizes = (size_t *)malloc(sizeof(size_t)*value_size);
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, value_size, max_work_item_sizes, NULL);
        printf("Max Work Item Sizes: %d \n\n", (int)(*max_work_item_sizes));
    }
}

int main(void) {
    int i;
    
    // Allocate space for vectors A, B and C
    float alpha = 1.0;
    float *A = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    float *B = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    float *C = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    for (i = 0; i < VECTOR_SIZE; i++)
    {
        A[i] = i;
        B[i] = VECTOR_SIZE - i;
        C[i] = 0;
    }
    
    cl_platform_id* platforms;
    cl_uint num_platforms;
    get_platforms(&platforms,num_platforms);
    cl_device_id* device_list;
    cl_uint num_devices;
    get_devices(platforms, num_platforms, &device_list, num_devices);
    cl_int clStatus;
    
    // Create one OpenCL context for each device in the platform
    cl_context context;
    context = clCreateContext(NULL, num_devices, device_list,
                              NULL, NULL, &clStatus);
    
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(
                                                          context, device_list[0], 0, &clStatus);
    
    // Create memory buffers on the device for each vector
    cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                    VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                    VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    
    // Copy the Buffer A and B to the device
    clStatus = clEnqueueWriteBuffer(command_queue, A_clmem,
                                    CL_TRUE, 0, VECTOR_SIZE * sizeof(float),
                                    A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(command_queue, B_clmem,
                                    CL_TRUE, 0, VECTOR_SIZE * sizeof(float),
                                    B, 0, NULL, NULL);
    
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
                                                   (const char **)&saxpy_kernel, NULL, &clStatus);
    
    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL,
                              NULL, NULL);
    
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "saxpy_kernel",
                                      &clStatus);
    
    // Set the arguments of the kernel
    clStatus = clSetKernelArg(kernel, 0, sizeof(float),
                              (void *)&alpha);
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem),
                              (void *)&A_clmem);
    clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem),
                              (void *)&B_clmem);
    
    clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem),
                              (void *)&C_clmem);
    
    // Execute the OpenCL kernel on the list
    size_t global_size = VECTOR_SIZE; // Process the entire lists
    size_t local_size = 64; // Number of work items that make up a workgroup
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1,
                                      NULL, &global_size, &local_size, 0, NULL, NULL);
    
    // Read the cl memory C_clmem on device to the host variable C
    clStatus = clEnqueueReadBuffer(command_queue, C_clmem,
                                   CL_TRUE, 0, VECTOR_SIZE * sizeof(float), C, 0, NULL, NULL);
    
    // Clean up and wait for all the comands to complete.
    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);
    
    // Display the result to the screen
    for (i = 0; i < VECTOR_SIZE; i++)
        printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);
    
    // Finally release all OpenCL allocated objects and host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(A_clmem);
    clStatus = clReleaseMemObject(B_clmem);
    clStatus = clReleaseMemObject(C_clmem);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    free(platforms);
    free(device_list);
    getchar();
    return 0;
}


