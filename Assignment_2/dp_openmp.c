#include <stdio.h>

#include <stdlib.h>
#include <string.h>

#include <time.h>
#include <sys/time.h>

#include "dp_openmp.h"

int main(int argc ,char* argv[]) {
	
	FILE *matrix_data;
	FILE *w;
	size_t size;
	size_t sizew;
	
	/* Initialize rows, cols, CUDA devices and threads from the user */
	unsigned int rows=atoi(argv[3]);
	unsigned int cols=atoi(argv[4]);
	int nprocs =atoi(argv[5]);
	
	printf("Rows= %d\n,Cols = %d\n",rows,cols);
	
	/*Host variable declaration */


	float* host_results = (float*) malloc(rows * sizeof(float)); 
	struct timeval starttime, endtime;
	clock_t start, end;
	float seconds = 0;
	unsigned int jobs; 
	unsigned long i;

	/*Kernel variable declaration */
	

        //size_t len = 0;
	float arr[rows][cols];
	float var ;
	int vrow =1;

	start = clock();

	/* Validation to check if the data file is readable */
	
	matrix_data = fopen(argv[1], "r");
	w = fopen(argv[2],"r");
	
	if (matrix_data == NULL)
	{
    		printf("Cannot Open the data ");
		return 0;
	}
	if (w == NULL)
	{
    		printf("Cannot Open the vector");
		return 0;
	}
	
	size = (size_t)((size_t)rows * (size_t)cols);
	sizew = (size_t)((size_t)vrow*(size_t)cols);

	printf("Size of the data = %lu\n",size);
	printf("Size of the vector = %lu\n",sizew);

	fflush(stdout);

	float *dataT = (float*)malloc((size)*sizeof(float));
	float *dataV = (float*)malloc((sizew)*sizeof(float));

	if(dataT == NULL) {
	        printf("ERROR: Memory for data not allocated.\n");
	}
	if(dataV == NULL) {
	        printf("ERROR: Memory for vector not allocated.\n");
	}
	
        gettimeofday(&starttime, NULL);
	int j = 0;

    /* Transfer the Data from the file to CPU Memory */
	

        for (i =0; i< rows;i++){
		for(j=0; j<cols ; j++){
			fscanf(matrix_data,"%f",&var);
                        arr[i][j]=var;
		}
	}
	for (i =0;i<cols;i++){
		for(j= 0; j<rows; j++){
			dataT[rows*i+j]= arr[j][i];
	}
	}		

		for (j=0;j<cols;j++){
			fscanf(w,"%f",&dataV[j]);
		}
   
	fclose(matrix_data);
	fclose(w);
		printf("Read Data\n");
        fflush(stdout);

        gettimeofday(&endtime, NULL);
        seconds+=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);

        printf("time to read data = %f\n", seconds);
    
	jobs =(unsigned int) ((rows+nprocs-1)/nprocs);
	

        gettimeofday(&starttime, NULL);
	
	printf("jobs=%d\n",jobs);
	
	kernel(rows,cols,dataT,dataV,host_results,jobs);

	gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time for kernel=%f\n", seconds);
		
	/* Copy the results back in host */
	
	
	printf("Output of dot product is \n");
	printf("\n");
	
	for(i = 0; i < rows; i++) {
		printf("%f ", host_results[i]);
	}
	printf("\n");


	end = clock();
	seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Total time = %f\n", seconds);

	return 0;

}
