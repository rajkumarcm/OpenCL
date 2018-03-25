//
//  OpenCL.h
//  SampleOpenCL
//
//  Created by Rajkumar Conjeevaram Mohan on 23/03/2018.
//  Copyright © 2018 Rajkumar Conjeevaram Mohan. All rights reserved.
//

#ifndef OpenCL_h
#define OpenCL_h
#endif
#ifdef __APPLE__
   #include <OpenCL/cl.h>
#else
   #include <CL/cl.h>
#endif
#include <fstream>
typedef unsigned short int u_sint;

struct OpenCLSetup
{
    public:
    
        cl_platform_id* platforms;
        cl_device_id* device_list;
        cl_context context;
        cl_command_queue * command_queues;
        int * work_item_sizes;
    
    
        OpenCLSetup(u_sint num_command_queues);
        ~OpenCLSetup();
        void get_platforms(cl_platform_id** platforms, cl_uint& num_platforms);
        void get_devices(cl_platform_id* platforms, const cl_uint num_platforms,
                         cl_device_id** device_list, cl_uint& num_devices);
    private:
        u_sint n_command_queues;
};

OpenCLSetup::OpenCLSetup(u_sint num_command_queues)
{
    
    n_command_queues = num_command_queues;
    cl_uint num_platforms;
    get_platforms(&platforms,num_platforms);
    cl_uint num_devices;
    get_devices(platforms, num_platforms, &device_list, num_devices);
    work_item_sizes = new int[num_devices];
    
    if(num_command_queues > num_devices)
        throw "Number of command queues cannot exceed the number"
               "of devices available";
        
    if(num_command_queues <= 0)
        throw "Invalid num_command_queues value";
    
    cl_int clStatus;
    // Create one OpenCL context for each device in the platform
    context = clCreateContext(NULL, num_devices, device_list,
                              NULL, NULL, &clStatus);
    
    // Create a command queue
    command_queues = new cl_command_queue[num_command_queues];
    for(u_sint i = 0; i < num_command_queues; i++)
    {
        command_queues[i] = clCreateCommandQueue(context,
                                                 device_list[i],
                                                 0,
                                                 &clStatus);
    }
    
}

OpenCLSetup::~OpenCLSetup()
{
    cl_int clStatus;
    for(u_sint i = 0; i < n_command_queues; i++)
        clStatus = clReleaseCommandQueue(command_queues[i]);
    clStatus = clReleaseContext(context);
    free(platforms);
    free(device_list);
}

void OpenCLSetup::get_platforms(cl_platform_id** platforms, cl_uint& num_platforms)
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

void OpenCLSetup::get_devices(cl_platform_id* platforms, const cl_uint num_platforms,
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
        char * value;
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
        
        cl_ulong * global_size;
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, 0, NULL, &value_size);
        global_size = (cl_ulong *)malloc(sizeof(cl_ulong)*value_size);
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, value_size, global_size, NULL);
        
        double temp_size = ((*global_size / 1024) / 1024) / 1024;
        printf("Global Memory: %.2fGB \n",temp_size);
        
        cl_uint * clock_freq;
        clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, 0, NULL, &value_size);
        clock_freq = (cl_uint *)malloc(sizeof(cl_uint)*value_size);
        clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, value_size, clock_freq, NULL);
        printf("Max Clock Frequency: %dMHz \n", *clock_freq);
        
        cl_uint * compute_units;
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, 0, NULL, &value_size);
        compute_units = (cl_uint*)malloc(sizeof(cl_uint)*value_size);
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, value_size, compute_units, NULL);
        printf("Max Compute Units: %d \n", *compute_units);
        
        size_t * max_work_group_size;
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, NULL, &value_size);
        max_work_group_size = (size_t *) malloc(sizeof(size_t)*value_size);
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, value_size, max_work_group_size, NULL);
        printf("Max Work Group Size: %d \n", (int)(*max_work_group_size));
        
        size_t * max_work_item_sizes;
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &value_size);
        max_work_item_sizes = (size_t *)malloc(sizeof(size_t)*value_size);
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, value_size, max_work_item_sizes, NULL);
        printf("Max Work Item Sizes: %d \n\n", (int)(*max_work_item_sizes));
//        work_item_sizes[i] = (int)(*max_work_item_sizes);
    }
}
