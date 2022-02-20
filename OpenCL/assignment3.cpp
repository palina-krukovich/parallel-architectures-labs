#include <stdio.h>
#include <iostream>
#include <sys/stat.h>
#include <cassert>
#include <cstring>
#include <chrono>
#include <sstream>
#include <fstream>
#include <cmath>
#include <time.h>
#include "CL/cl.h"
#include "conf.hpp"

#define WARP_SIZE				32
#define MAX_PLATFORMS			8
#define MAX_DEVICES				8
#define TARGET_PLATFORM			"NVIDIA CUDA\0"
#define TARGET_DEVICE			"Tesla K20m\0"
#define NUM_DEVICES_USED		1


// settings
#define SOURCE_CODE_FILE_PATH	"assignment3.cl"


#if (WORK_GROUP_SIZE_X * WORK_GROUP_SIZE_Y) < (WARP_SIZE)
#warning	"work group size should be bigger than warp size !!!!"
#endif

#if (WORK_SIZE_X * WORK_SIZE_Y) % (WARP_SIZE)
#warning	"work group size should be divisible to warp size !!!!"
#endif

static_assert(WORK_GROUP_SIZE_X > 0, "WORK_GROUP_SIZE_X must be bigger than 0");
static_assert(WORK_GROUP_SIZE_Y > 0, "WORK_GROUP_SIZE_Y must be bigger than 0");
static_assert(HEIGHT > 0, "HEIGHT must be bigger than 0");
static_assert(WIDTH > 0, "WIDTH must be bigger than 0");
static_assert(NUM_OF_ITERS > 0, "NUM_OF_ITERS must be bigger than 0");

char *read_source_code(const char *path, size_t *source_size) 
{
	FILE *f;
	struct stat st;
	char *source = NULL;
	
	if ((f = fopen(path, "r")) == NULL) 
	{
		return source;
	}

	fstat(fileno (f), &st);

	*source_size = st.st_size;
	source = (char*)malloc(*source_size);
	if (!source)
	{
		return source;
	}

	if (fread(source, sizeof(source[0]), *source_size, f) != *source_size)
	{
		free(source);
		source = NULL;
	}	

	return source;
}

void base_init(float *r, float *g, float *b)
/* sequential init function */
{
	for (int i = 0; i < HEIGHT; ++i)
	{
		for (int j = 0; j < WIDTH; ++j)
		{
			g[i*WIDTH+j] = 1.00;
			r[i*WIDTH+j] = (float)j/((float)i+1.00);
			b[i*WIDTH+j] = (float)i / ((float)j + 1.00);
		}
	}
}

void base_solution(float *r, float *g, float *b)
/* sequential solution function */
{
	for (int t = 0; t < NUM_OF_ITERS; ++t)
	{	
		//each row - beware first row and last row not to be updated therefore from 1...8190
		for(int i = 1; i < HEIGHT - 1; ++i)
		{
			//each column
			for(int j = 0; j < WIDTH; ++j)
			{
				//only matrix k=1 is updated
				g[i*WIDTH+j] = g[i*WIDTH+j] + (1 / (sqrt(b[i*WIDTH+j+WIDTH] + r[i*WIDTH+j-WIDTH])));
			}
		}
	}
}

int main(int argc, char* argv[])
{
	cl_platform_id platforms[MAX_PLATFORMS];
	cl_device_id devices[MAX_PLATFORMS];
	cl_platform_id target_platform_uid = 0;
	cl_int cl_ret_results;
	cl_device_id target_device_uid = 0;
	cl_uint num_platforms;
	cl_uint num_devices;
	cl_program program;
	const char *source;
	size_t source_size;
	cl_context context;
	cl_kernel kernel;
	cl_command_queue command_queue;
	cl_mem rb;
	cl_mem gb;
	cl_mem bb;
	double time_sec;
	clock_t start, end;

	float *r;
	float *g;
	float *b;

	float *r_check;
	float *g_check;
	float *b_check;

	/* 2d range of HEIGHTx(WIDTH / 2) */
	/* HEIGHT - 2 since we don't need upper and lower part of matrix*/
	const size_t ndrange[] = {HEIGHT - 2, WIDTH / 2};
	const size_t work_group[] = {WORK_GROUP_SIZE_Y, WORK_GROUP_SIZE_X};
//=================================== FIND PLATFORM ===================================

	// Print all available platforms and choose nvidia
	cl_ret_results = clGetPlatformIDs(sizeof(platforms) / sizeof(platforms[0]), platforms, &num_platforms); 
	assert(cl_ret_results == CL_SUCCESS);

	std::cout << "Number of available platforms: " << num_platforms << std::endl;
	for (int i = 0; i < num_platforms; ++i)
	{
		char *name;
		size_t size;

        cl_ret_results = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &size);
		assert(cl_ret_results == CL_SUCCESS);

        name = (char*)malloc(size);

        cl_ret_results = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, size, name, NULL);
		assert(cl_ret_results == CL_SUCCESS);

		std::cout << i << ": " << name << " (uid: " << platforms[i] << ")" << std::endl;

		if (!strcmp(TARGET_PLATFORM, name))
		{
			target_platform_uid = platforms[i];
		}
	}

	assert(target_platform_uid != 0);

//=================================== FIND DEVICES ==================================

	// Print all available devices and choose k20m

	cl_ret_results = clGetDeviceIDs(target_platform_uid, CL_DEVICE_TYPE_GPU, MAX_DEVICES, devices, &num_devices);
	assert(cl_ret_results == CL_SUCCESS);

	std::cout << "Number of devices: " << num_devices << std::endl;

	for (int i = 0; i < num_devices; ++i)
	{
		char *name;
		size_t size;

        cl_ret_results = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &size);
		assert(cl_ret_results == CL_SUCCESS);

        name = (char*)malloc(size);

        cl_ret_results = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, size, name, NULL);
		assert(cl_ret_results == CL_SUCCESS);

		std::cout << i << ": " << name << " (uid: " << devices[i] << ")" << std::endl;

		if (!strcmp(TARGET_DEVICE, name))
		{
			target_device_uid = devices[i];
		}
	}

	assert(target_device_uid != 0);

//=================================== CREATE CONTEXT ==================================

	context = clCreateContext(NULL, NUM_DEVICES_USED, &target_device_uid, NULL, NULL, &cl_ret_results);
	assert(cl_ret_results == CL_SUCCESS);

//=================================== READ AND BUILD PROGRAM ==================================


	source = read_source_code(SOURCE_CODE_FILE_PATH, &source_size);

	program = clCreateProgramWithSource(context, 1, &source, &source_size, &cl_ret_results);
	assert(cl_ret_results == CL_SUCCESS);

	/* use optimizations from lecture */
	cl_ret_results = clBuildProgram(program, 1, &target_device_uid, "-cl-no-signed-zeros -cl-mad-enable -cl-unsafe-math-optimizations", NULL, NULL);
	assert(cl_ret_results == CL_SUCCESS);

//=================================== Arrays init ==================================

	size_t sz = sizeof(float) * HEIGHT * WIDTH;
	r = (float*)malloc(sz);
	g = (float*)malloc(sz);
	b = (float*)malloc(sz);

	r_check = (float*)malloc(sz);
	g_check = (float*)malloc(sz);
	b_check = (float*)malloc(sz);

//=================================== CHECK SOLVE ==================================
	/* run init fuinctions and sequential solution */
	base_init(r, g, b);
	base_init(r_check, g_check, b_check);

	start = clock();
	base_solution(r_check, g_check, b_check);
	end = clock();
	time_sec = ((double) (end - start)) / CLOCKS_PER_SEC;
	std::cout << "Ordinary solution takes: " << time_sec << " sec" << std::endl;
//=================================== MAKE BUFFERS ==================================
	rb = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, HEIGHT * WIDTH * sizeof(float), (void*)r, &cl_ret_results);
	assert(cl_ret_results == CL_SUCCESS);

	bb = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, HEIGHT * WIDTH * sizeof(float), (void*)b, &cl_ret_results);
	assert(cl_ret_results == CL_SUCCESS);

	gb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, HEIGHT * WIDTH * sizeof(float), (void*)g, &cl_ret_results);
	assert(cl_ret_results == CL_SUCCESS);

//=================================== KERNEL AND COMMAND QUEUE ==================================

	kernel = clCreateKernel(program, "opencl_solution", &cl_ret_results);
	assert(cl_ret_results == CL_SUCCESS);

	command_queue = clCreateCommandQueue(context, target_device_uid, 0, &cl_ret_results);
	assert(cl_ret_results == CL_SUCCESS);
//=================================== MAKE ARGS ==================================

	int w = WIDTH;
	int h = HEIGHT;
	
	cl_ret_results = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&rb);
	assert(cl_ret_results == CL_SUCCESS);
	
	cl_ret_results = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&gb);
	assert(cl_ret_results == CL_SUCCESS);

	cl_ret_results = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bb);
	assert(cl_ret_results == CL_SUCCESS);
	
	cl_ret_results = clSetKernelArg(kernel, 3, sizeof(int), (void *)&h);
	assert(cl_ret_results == CL_SUCCESS);

	cl_ret_results = clSetKernelArg(kernel, 4, sizeof(int), (void *)&w);
	assert(cl_ret_results == CL_SUCCESS);

//=================================== MAIN LOGIC ==================================
	// fill queue with tasks
	for (int t = 0; t < NUM_OF_ITERS; ++t) 
	{
		cl_ret_results = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
											ndrange, work_group, 0, NULL, NULL);
		assert(cl_ret_results == CL_SUCCESS);
	}

	start = clock();

	clFinish(command_queue);

	end = clock();
	time_sec = ((double) (end - start)) / CLOCKS_PER_SEC;
	std::cout << "OpenCL solution takes: " << time_sec << " sec" << std::endl;
	// read results

	cl_ret_results = clEnqueueReadBuffer(command_queue, gb, CL_TRUE, 0, HEIGHT * WIDTH * sizeof(float), g, 0, NULL, NULL);
	assert(cl_ret_results == CL_SUCCESS);


	// if macro defined  - print matrix to console
#if defined(PRINT_MATRIX)
	for (int i = 0; i < HEIGHT; ++i)
	{
		for (int j = 0; j < WIDTH; ++j)
		{
			std::cout << g[i*WIDTH+j] << " ";
		}
		std::cout << std::endl;
	}
#endif /* PRINT_MATRIX */

	/* check results with sequential solution */
	for (int i = 0; i < HEIGHT * WIDTH; ++i)
	{
		assert(abs(r[i] - r_check[i]) < 0.001);
		assert(abs(g[i] - g_check[i]) < 0.001);
		assert(abs(b[i] - b_check[i]) < 0.001);
	}
	std::cout << "Check success" << std::endl;

//=================================== FREE MEMORY ==================================

	free(r);
	free(b);
	free(g);

	free(r_check);
	free(b_check);
	free(g_check);
	return 0;
}

