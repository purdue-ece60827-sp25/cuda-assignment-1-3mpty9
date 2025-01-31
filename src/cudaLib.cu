
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < size)
		y[idx] = scale * x[idx] + y[idx];

}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	size_t vectorBytes = vectorSize * sizeof(float);
	
	//	Allocate input vectors h_x and h_y in host memory
	float* h_a = (float*)malloc(vectorBytes);
	float* h_b = (float*)malloc(vectorBytes);
	float* h_c = (float*)malloc(vectorBytes);

	

	if (!h_a || !h_b || !h_c ) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	//	Initialize input vectors
	vectorInit(h_a, vectorSize);
	vectorInit(h_b, vectorSize);
	memcpy(h_c, h_b, vectorBytes);

	float scale = 2.0f;

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", h_a[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", h_b[i]);
		}
		printf(" ... }\n");
	#endif

	//	Allocate vectors in devie memory
	float* d_a;
	cudaMalloc(&d_a, vectorBytes);
	float* d_b;
	cudaMalloc(&d_b, vectorBytes);

	// Copy vectors from host memory to device memory
	cudaMemcpy(d_a, h_a, vectorBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, vectorBytes, cudaMemcpyHostToDevice);

	// Invoke kernel
	int threadsPerBlock = 256;
    int blocksPerGrid =
            (vectorSize + threadsPerBlock - 1) / threadsPerBlock;
    saxpy_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, scale, vectorSize);


	// Copy result from device memory to host memory
    // h_C contains the result in host memory
	cudaMemcpy(h_c, d_b, vectorBytes, cudaMemcpyDeviceToHost);

	#ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", h_c[i]);
		}
		printf(" ... }\n");
	#endif

	int errorCount = verifyVector(h_a, h_b, h_c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	// Free device memory
	cudaFree(d_a);
    cudaFree(d_b);

	// Free host memory
    free(h_a);
    free(h_b);
	free(h_c);

	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx >= pSumSize) return;

	curandState_t rng;
	curand_init(clock64(), idx, 0, &rng);

	uint64_t hitCount = 0;

	for(uint64_t i = 0; i < sampleSize; i++){
		float x = curand_uniform(&rng);
		float y = curand_uniform(&rng);
		if(int(x * x + y * y) == 0){
			++ hitCount;
		}
	}
	
	pSums[idx] = hitCount;

}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= reduceSize) return;

    uint64_t sum = 0;
    for (uint64_t i = idx; i < pSumSize; i += reduceSize) {
        sum += pSums[i];
    }
    totals[idx] = sum;
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	uint64_t *d_pSums;
	size_t pSumsBytes = generateThreadCount * sizeof(uint64_t);
	uint64_t *d_totals;
	size_t totalsBytes = generateThreadCount * sizeof(uint64_t);
	cudaMalloc(&d_pSums, pSumsBytes);
	cudaMalloc(&d_totals, totalsBytes);

	generatePoints<<<(generateThreadCount + 255) / 256, 256>>>(d_pSums, generateThreadCount, sampleSize);
	reduceCounts<<<(reduceThreadCount + 255) / 256, 256>>>(d_pSums, d_totals, generateThreadCount, reduceSize);

	uint64_t *h_totals = (uint64_t*)malloc(totalsBytes);
	cudaMemcpy(h_totals, d_totals, totalsBytes, cudaMemcpyDeviceToHost);

	uint64_t totalHitCount = 0;
	for (int i = 0; i < reduceThreadCount; ++i) {
		totalHitCount += h_totals[i];
	}

	// Calculate the approximate value of pi
	approxPi = ((double)totalHitCount / sampleSize) / generateThreadCount;
	approxPi = approxPi * 4.0f;
	// Free allocated memory
	cudaFree(d_pSums);
	cudaFree(d_totals);
	free(h_totals);

	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}
