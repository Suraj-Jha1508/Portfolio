#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "dp.h"

__global__ void kernel(unsigned int rows, unsigned int cols , float* ddata,float* vdata ,float *results){
	
	int i;
        float sum =0;
	int tid  = threadIdx.x + blockIdx.x * blockDim.x;
	
	for(i =0; i<cols ;i++ )
	{
		sum+= ddata[i*rows+tid]*vdata[i];		
	}
	
	results[tid] = sum;
	
}
