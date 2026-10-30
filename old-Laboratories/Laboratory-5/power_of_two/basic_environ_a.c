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
	  
	// check error, in such a case, it exits

	void cl_error(cl_int code, const char *string){
		if (code != CL_SUCCESS){
			printf("%d - %s\n", code, string);
			exit(-1);
		}
	}
	////////////////////////////////////////////////////////////////////////////////

	int main(int argc, char** argv)
	{
	  int err;                            	// error code returned from api calls
	  size_t t_buf = 50;			// size of str_buffer
	  char str_buffer[t_buf];		// auxiliary buffer	
	  size_t e_buf;				// effective size of str_buffer in use
	  size_t program_Size;			// size of the opencl program
		
		  
	  size_t global_size;                      	// global domain size for our calculation
	  size_t local_size;                       	// local domain size for our calculation

	  const cl_uint num_platforms_ids = 10;				// max of allocatable platforms
	  cl_platform_id platforms_ids[num_platforms_ids];		// array of platforms
	  cl_uint n_platforms;						// effective number of platforms in use
	  const cl_uint num_devices_ids = 10;				// max of allocatable devices
	  cl_device_id devices_ids[num_platforms_ids][num_devices_ids];	// array of devices
	  cl_uint n_devices[num_platforms_ids];				// effective number of devices in use for each platform
		
	  cl_device_id device_id;             				// compute device id 
	  cl_context context;                 				// compute context
	  cl_command_queue command_queue;     				// compute command queue
	  cl_program program;                 				// compute program
	  cl_kernel kernel;                   				// compute kernel
		
	  size_t count = 1024;
	  float in_host_object[count];
	  float out_host_object[count];

	  for (int i = 0; i < count; i++){
		in_host_object[i] = (float) i;
	  }


	  // 1. Scan the available platforms:
	  err = clGetPlatformIDs (num_platforms_ids, platforms_ids, &n_platforms);
	  cl_error(err, "Error: Failed to Scan for Platforms IDs");
	  printf("Number of available platforms: %d\n\n", n_platforms);

	  for (int i = 0; i < n_platforms; i++ ){
		err= clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_NAME, t_buf, str_buffer, &e_buf);
		cl_error (err, "Error: Failed to get info of the platform\n");
		printf( "\t[%d]-Platform Name: %s\n", i, str_buffer);
	  }
	  printf("\n");
	  // ***Task***: print on the screen the name, host_timer_resolution, vendor, versionm, ...
		
	  // 2. Scan for devices in each platform
	  for (int i = 0; i < n_platforms; i++ ){
		err = clGetDeviceIDs(platforms_ids[i], CL_DEVICE_TYPE_ALL, num_devices_ids, devices_ids[i], &(n_devices[i]));
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
	  cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[0], 0};
	  context = clCreateContext(properties, 1, devices_ids[0], NULL, NULL, &err);
	  cl_error(err, "Failed to create a compute context\n");

	  // 4. Create a command queue
	  cl_command_queue_properties proprt[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	  command_queue = clCreateCommandQueueWithProperties(context, devices_ids[0][0], proprt, &err);
	  cl_error(err, "Failed to create a command queue\n");

	  // 5. Read an OpenCL program from the file kernel.cl
	  // Calculate size of the file
	  FILE *fileHandler = fopen("kernel.cl", "r");
	  fseek(fileHandler, 0, SEEK_END);
	  size_t fileSize = ftell(fileHandler);
	  rewind(fileHandler);

	  // read kernel source into buffer
	  char * sourceCode = (char*) malloc(fileSize + 1);
	  sourceCode[fileSize] = '\0';
	  fread(sourceCode, sizeof(char), fileSize, fileHandler);
	  fclose(fileHandler);

	  // create program from buffer
	  program = clCreateProgramWithSource(context, 1, (const char**) &sourceCode, &fileSize, &err);
	  cl_error(err, "Failed to create program with source\n");
	  free(sourceCode);

	  // read kernel source back in from program to check
	  size_t kernelSourceSize;
	  clGetProgramInfo(program, CL_PROGRAM_SOURCE, 0, NULL, &kernelSourceSize);
	  char *kernelSource = (char*) malloc(kernelSourceSize);
	  clGetProgramInfo(program, CL_PROGRAM_SOURCE, kernelSourceSize, kernelSource, NULL);
	  printf("nKernel source:\n\n%s\n", kernelSource);
	  free(kernelSource);

	  // Build the executable and check errors
	  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	  if (err != CL_SUCCESS){
		size_t len;
		char buffer[2048];

		printf("Error: Some error at building process.\n");
		clGetProgramBuildInfo(program, devices_ids[0][0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(-1);
	  }

	  // Create a compute kernel with the program we want to run
	  kernel = clCreateKernel(program, "pow_of_two", &err);
	  cl_error(err, "Failed to create kernel from the program\n");

	  // Create OpenCL buffer visible to the OpenCl runtime
	  cl_mem in_device_object  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, &err);
	  cl_error(err, "Failed to create memory buffer at device\n");
	  cl_mem out_device_object = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, &err);
	  cl_error(err, "Failed to create memory buffer at device\n");

	  // Write date into the memory object 
	  err = clEnqueueWriteBuffer(command_queue, in_device_object, CL_TRUE, 0, sizeof(float) * count, in_host_object, 0, NULL, NULL);
	  cl_error(err, "Failed to enqueue a write command\n");

	  // Set the arguments to the kernel
	  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_device_object);
	  cl_error(err, "Failed to set argument 0\n");
	  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_device_object);
	  cl_error(err, "Failed to set argument 1\n");
	  err = clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
	  cl_error(err, "Failed to set argument 2\n");


	  // Launch Kernel
	  local_size = 128;
	  global_size = count;
	  err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	  cl_error(err, "Failed to launch kernel to the device\n");

  // Read data form device memory back to host memory
  err = clEnqueueReadBuffer(command_queue, out_device_object, CL_TRUE, 0, sizeof(float) * count, out_host_object, 0, NULL, NULL);
  cl_error(err, "Failed to enqueue a read command\n");

  printf("Square of %f is: %f\n", in_host_object[324], out_host_object[324]);

  clReleaseMemObject(in_device_object);
  clReleaseMemObject(out_device_object);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);



  return 0;
}

