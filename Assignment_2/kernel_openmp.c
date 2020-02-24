#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dp_openmp.h"

void kernel(unsigned int rows, unsigned int cols , float* ddata,float* vdata ,float *results,unsigned int jobs){
	
	int i,j,stop;
        float sum =0;
	//int tid  = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = omp_get_thread_num();
	
        if((tid+1)*jobs > rows) stop=rows;
        else stop = (tid+1)*jobs;
     
	printf("thread id=%d, start=%d, stop=%d\n", tid, tid*jobs, stop);
	for (j = tid*jobs; j < stop; j++) { 
	    sum=0;
	    for(i =0; i<rows ;i++ )
	    {
		sum+= ddata[i*rows+j]*vdata[i];		
	    }
	
	results[j] = sum;
	}
}
