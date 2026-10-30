////////////////////////////////////////////////////////////////////
//File: basic_environ.c
//
//Description: base file for environment exercises with openCL
//
// 
////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

// ABOUT ERRORS
// Remmember to check error codes from every OpenCL API call
// Info about error codes can be found at the reference manual of OpenCL
// At the following url, you can find name for each error code
//  https://gist.github.com/bmount/4a7144ce801e5569a0b6
//  https://streamhpc.com/blog/2013-04-28/opencl-error-codes/
// Following function checks errors, and in such a case, it prints the code, the string and it exits

void cl_error(cl_int code, const char *string){
    if (code != CL_SUCCESS){
        printf("%d - %s\n", code, string);
        exit(-1);
    }
}
////////////////////////////////////////////////////////////////////////////////


/* ATTENTION: While prgramming in OpenCL it is a good idea to keep the reference manuals handy:
 * https://bashbaug.github.io/OpenCL-Docs/pdf/OpenCL_API.pdf
 * https://www.khronos.org/files/opencl-1-2-quick-reference-card.pdf (summary of OpenCL API)
 * https://www.khronos.org/assets/uploads/developers/presentations/opencl20-quick-reference-card.pdf
 */


int main(int argc, char** argv)
{
  int err;                              // error code returned from api calls
  size_t t_buf = 50;            // size of str_buffer
  char str_buffer[t_buf];       // auxiliary buffer 
  size_t e_buf;             // effective size of str_buffer in use
        
  size_t global_size;                       // global domain size for our calculation
  size_t local_size;                        // local domain size for our calculation

  const cl_uint num_platforms_ids = 10;             // max of allocatable platforms
  cl_platform_id platforms_ids[num_platforms_ids];      // array of platforms
  cl_uint n_platforms;                      // effective number of platforms in use
  const cl_uint num_devices_ids = 10;               // max of allocatable devices
  cl_device_id devices_ids[num_platforms_ids][num_devices_ids]; // array of devices
  cl_uint n_devices[num_platforms_ids];             // effective number of devices in use for each platform
    
  cl_device_id device_id;                           // compute device id 
  cl_context context;                               // compute context
  cl_command_queue command_queue;                   // compute command queue
    

  // 1. Scan the available platforms:
  err = clGetPlatformIDs (num_platforms_ids, platforms_ids, &n_platforms);
  cl_error(err, "Error: Failed to Scan for Platforms IDs");
  printf("Number of available platforms: %d\n\n", n_platforms);

  for (int i = 0; i < n_platforms; i++ ){
    err= clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_NAME, /***???***/);
    cl_error (err, "Error: Failed to get info of the platform\n");
    printf( "\t[%d]-Platform Name: %s\n", i, str_buffer);
  }
  printf("\n");
  // ***Task***: print on the screen the name, host_timer_resolution, vendor, versionm, ...
    
  // 2. Scan for devices in each platform
  for (int i = 0; i < n_platforms; i++ ){
    err = clGetDeviceIDs( /***???***/, num_devices_ids, devices_ids[i], &(n_devices[i]));
    cl_error(err, "Error: Failed to Scan for Devices IDs");
    printf("\t[%d]-Platform. Number of available devices: %d\n", i, n_devices[i]);

    for(int j = 0; j < n_devices[i]; j++){
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_NAME, sizeof(str_buffer), &str_buffer, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device name");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_NAME: %s\n", i, j,str_buffer);

      cl_uint max_compute_units_available;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_available), &max_compute_units_available, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device max compute units available");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_MAX_COMPUTE_UNITS: %d\n\n", i, j, max_compute_units_available);
    }
  } 
  // ***Task***: print on the screen the cache size, global mem size, local memsize, max work group size, profiling timer resolution and ... of each device



  // 3. Create a context, with a device
  cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[/***???***/], 0};
  context = clCreateContext(properties, /***???***/, NULL, NULL, &err);
  cl_error(err, "Failed to create a compute context\n");

  // 4. Create a command queue
  cl_command_queue_properties proprt[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
  command_queue = clCreateCommandQueueWithProperties( /***???***/, proprt, &err);
  cl_error(err, "Failed to create a command queue\n");

  /* It is still missing the runtime part of the OpenCL program: createBuffers, createProgram, createKernel, setKernelArg, ... */
  
  return 0;
}

