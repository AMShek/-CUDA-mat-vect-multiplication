#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#define BLOCK_DIM 10

//一次核函数调用计算一个结果C中的元素
void __global__ MVMulCUDA(float *A,  float *B, float *C, int rowSize, int columnSize, int wA){
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	uint t_id = blockDim.x * bx+ tx;

    if(rowSize <= t_id)
		return;
    
	float Csub = 0;
    for(int i = 0; i < columnSize; i++){
		Csub += A[t_id * wA + i] * B[i];
    }
    C[t_id] = Csub;
}
__global__ void transpose(float *odata, float *idata, int width, int height){
    __shared__ float block[(BLOCK_DIM+1)*BLOCK_DIM];
    unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
    unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
    unsigned int xIndex = xBlock + threadIdx.x;
    unsigned int yIndex = yBlock + threadIdx.y;
    unsigned int index_out, index_transpose;
    if (xIndex < width && yIndex < height)
    {
        unsigned int index_in = __mul24(width, yIndex) + xIndex;
        unsigned int index_block = __mul24(threadIdx.y, BLOCK_DIM+1) + threadIdx.x;
        block[index_block] = idata[index_in];
        index_transpose = __mul24(threadIdx.x, BLOCK_DIM+1) + threadIdx.y;
        index_out = __mul24(height, xBlock + threadIdx.y) + yBlock + threadIdx.x;
    }
    __syncthreads();
    if (xIndex < width && yIndex < height)
        odata[index_out] = block[index_transpose];
}
void ConstantInit_A(float *data, int w, int h) {
	//row
	for (int i = 0; i < h; i++) {
		//column
		for (int j = 0; j<w; j++) {
			data[i*w + j] = i - 0.1*j + 1;
		}
	}
}

void ConstantInit_B(float *data, int h) {
	//row
	for (int i = 0; i < h; i++) {
		data[i] = log(sqrt(i*i - i + 1));
	}
}
void MatrixTranspose(float* M, float* V, int width, int height){
    int size = width*height*sizeof(float);
    float* Md, *Vd;
    cudaMalloc(&Md, size);
    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    cudaMalloc(&Vd, size);
    dim3 dimGrid(width/10, height/10);
    dim3 dimBlock(10, 10);

    float time_elapsed=0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start,0);   
    transpose<<<dimGrid, dimBlock>>>(Vd, Md, width, height);
    cudaEventRecord( stop,0);   
    cudaEventSynchronize(start);   
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&time_elapsed,start,stop);  

    cudaMemcpy(V, Vd, size, cudaMemcpyDeviceToHost);

    //Free device matrices
    cudaFree(Md);
    cudaFree(Vd);
    printf("GPU running time for transposition：%f12 msec\n",time_elapsed);
}

void MVMul(float* A, float* B, float* C, int row, int column){
	int wA = 10;
	/*
	int width = column * sizeof(float);
	int	height = row;
	int columnSize = column * sizeof(float);
	int rowSize = row * sizeof(float);
	*/
	int size_A = row * column;
	int mem_size_A = sizeof(float)*size_A;
	float *h_A = (float*)malloc(mem_size_A);

	int size_B = row * 1;
	int mem_size_B = sizeof(float) * size_B;
	float *h_B = (float*)malloc(mem_size_B);


	// Allocate device memory
    float* d_A, *d_B, *d_C;
    
	// Allocate host vector C
	dim3 dimsC(row, 1, 1);
	int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
	float *h_C = reinterpret_cast<float *>(malloc(mem_size_C));
	if (h_C == NULL) {
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	//Allocate device memory
	cudaMalloc(&d_A, mem_size_A);
	cudaMalloc(&d_B, mem_size_B);
	cudaMalloc(&d_C, mem_size_C);
	

	// copy host memory to device
	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    //setup the execution configuration
    int dimGrid = 10;
    int dimBlock = 1000;

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	cudaEventCreate(&start);

	cudaEvent_t stop;
	cudaEventCreate(&stop);

	// Record the start event
	cudaEventRecord(start, NULL);

    //Launch the device computeation threads
    MVMulCUDA<<<dimGrid, dimBlock>>>(d_A, d_B, d_C,row, column, wA/sizeof(float));
	
	// Record the stop event
	cudaEventRecord(stop, NULL);

	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);

	// Compute and print the performance
	printf("GPU running time= %.12f msec\n",msecTotal);

    //Read C from device
    cudaMemcpy(C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    //Free device matrices
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


int show(float* C, int rowSize, int columnSize, int showsize){
    int i, j;
    if(showsize < rowSize)
        rowSize = showsize;
    if(rowSize == 1)
        columnSize = 0;
    for(i=0; i<rowSize; i++){
        for(j=0; j<showsize; j++){
            printf("%f ", C[i*columnSize + j]);
        }
        printf("\n");
    }
    return 0;
}


int main(){

    //申请内存
    float* A;
    float* B;
    float* C;
	float* A_T;
    int i, j, rowSize = 10000, columnSize = 10000;
    int size = rowSize * columnSize;
    A = (float*)malloc(size * sizeof(float));
    B = (float*)malloc(columnSize * sizeof(float));
    C = (float*)malloc(rowSize * sizeof(float));
	A_T = (float*)malloc(size * sizeof(float));
	
	
	//Initial matrix A and vector B
	ConstantInit_A(A, columnSize, rowSize);
	ConstantInit_B(B, rowSize);
	
    struct timeval tvs,tve;

    //Calculate
    cudaDeviceReset(); 
    gettimeofday(&tvs,NULL); 
	MatrixTranspose(A, A_T, columnSize, rowSize);	
    MVMul(A, B, C, rowSize, columnSize);
    gettimeofday(&tve,NULL);
    cudaDeviceReset();    

    //Result examples
    double span = tve.tv_sec-tvs.tv_sec + (tve.tv_usec-tvs.tv_usec)/1000000.0;
    printf("CPU running time: %.12f sec\n",span);
    printf("Result examples:\n");
    show(C, 1, rowSize, 10);

    free(A);
    free(B);
    free(C);
    A = NULL;
    B = NULL;
    C = NULL;

    return 0;
}
