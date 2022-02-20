#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#include "conf.hpp"

__kernel void opencl_solution(__global const float *r, __global float *g, __global const float *b, const int h, const int w)
{
	const int i = get_global_id(0) + 1;
	const int j = get_global_id(1);
	const int local_id = get_local_id(0);
	float tmp;
	__local float r_local[WORK_GROUP_SIZE];
	__local float g_local[WORK_GROUP_SIZE];
	__local float b_local[WORK_GROUP_SIZE];

	
	const int offset = i * w + j;
	int pos = offset + 0;

	int rpos = pos - w;
	int bpos = pos + w;
	r_local[local_id] = r[rpos];
	g_local[local_id] = g[pos];
	b_local[local_id] = b[bpos];
	tmp = native_rsqrt(b_local[local_id] + r_local[local_id]);
	g_local[local_id] += tmp;
	g[pos] = g_local[local_id];

	barrier(CLK_LOCAL_MEM_FENCE);
	pos = offset + w / 2;
	rpos = pos - w;
	bpos = pos + w;
	r_local[local_id] = r[rpos];
	g_local[local_id] = g[pos];
	b_local[local_id] = b[bpos];
	tmp = native_rsqrt(b_local[local_id] + r_local[local_id]);
	g_local[local_id] += tmp;
	g[pos] = g_local[local_id];
}
