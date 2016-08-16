#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>


/*

Row wise multiplication algorithm implemented in parallel. Accepts arbitrary numbers of equivalent size.
Both times will be printed to see the benefits of paralle computation for the large integer and answeres
will be verified based upon comparison of both algorithms (pass/fail). Regular row wise multiplication is
used and the carries for each product multiplication is dealt with by a sequential carry adder. The
multiplication algorithm is beneficially in the use of RSA encryption or Diffie Hellman key exchange. The algorithm could be implemented in other applications were large amounts of multiplication can be used in parallel to reduce computation time.

*/
__global__ void get_products(unsigned char a[], unsigned char b[], unsigned int accumulator[], unsigned int n);

int main(int argc, char *argv[]) {	

	if (argc != 3) {
		printf("usage: ./a.out N ThreadsPerBlock\n");
		exit(1);
	}
	printf("Version1, n = %s, threads = %s\n", argv[1], argv[2]);

	unsigned int n = atoi(argv[1]);
	unsigned int threads = atoi(argv[2]);

	unsigned char *p =  (unsigned char *) malloc(n);
	unsigned char *q =  (unsigned char *) malloc(n);


	//replace with the ability to read in file
	int t = 0;
	unsigned char hex;
	while(t < n) {
		hex = (unsigned char) (rand() % 255) + 1;
		p[t] = hex;
		t++;
	}

	t = 0;
	while(t < n) {
		hex = (unsigned char) (rand() % 255) + 1;
		q[t] = hex;
		t++;
	}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Row wise GPU version
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	unsigned char *d_A;
	cudaMalloc(&d_A, n);
	unsigned char *d_B;
	cudaMalloc(&d_B, n);
	unsigned int *d_C;
	cudaMalloc(&d_C, 2*n*sizeof(unsigned int));

	cudaMemcpy(d_A, p, n, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_B, q, n, cudaMemcpyHostToDevice);
	cudaMemset(d_C, 0, 2*n*sizeof(unsigned int));		

	dim3 blocksPerGrid(n/threads);
	dim3 threadsPerBlock(threads);

	cudaError_t error;
	cudaEvent_t start;

	error = cudaEventCreate(&start);
	if(error != cudaSuccess)
		printf("error\n");
	
	cudaEvent_t stop;
	error = cudaEventCreate(&stop);
	if(error != cudaSuccess)
		printf("error\n");

	error = cudaEventRecord(start, NULL);


	//call kernel to multiply a * b = c where a and b are of size n
	get_products<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

	//compute final answer with sequential adder
	unsigned char *final = (unsigned char *) malloc(2*n);
	memset(final, 0, 2*n);
	unsigned int *transfer = (unsigned int *) malloc(2*n*sizeof(unsigned int));

	//copy result of multiplication to cpu copy to calculate carries
	cudaMemcpy(transfer, d_C, 2*n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int index = 0;
	while(index < 2*n) {
		//cast to character and add to index of final result
		final[index] = (unsigned char) transfer[index];

		//collect the other three bytes and add to the next sequential 
		//integer index
		transfer[index + 1] += (unsigned int) (transfer[index]>>8);
		index++;
	}



	error = cudaEventRecord(stop, NULL);

	error = cudaEventSynchronize(stop);

	if(error != cudaSuccess)
		printf("error\n");

	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);// return is miliseconds

	printf("GPU time: %.6f\n", msecTotal / 1000);


///////////////////////////////////////////////////////////////////////////////////////////////////
// Row Wise CPU for time comparison
//////////////////////////////////////////////////////////////////////////////////////////////////
	unsigned char *cpu_result = (unsigned char *) malloc(2*n);
	memset(cpu_result, 0, 2*n);

	unsigned int multiplicand_position;
	struct timeval cpu_start, cpu_end;
	struct timezone tzp;

	gettimeofday(&cpu_start, &tzp);

	//loop through n rows of products
	for(multiplicand_position = 0; multiplicand_position < n; multiplicand_position++) {

		unsigned int result_position = multiplicand_position;
		unsigned char result_carry = 0;
		unsigned short cpu_product = 0;
		unsigned int multiplier_position = 0;
		unsigned short cpu_sum;

		unsigned int loop = 0;
		//loop through n multipliers
		while(loop < n) {

			//calculate the product of ch * ch
			unsigned short cpu_sum;
			cpu_product = p[multiplier_position] * q[multiplicand_position];

			multiplier_position++;

			//calculate the sum of previous carry, current result index, and current product
			cpu_sum = (cpu_result[result_position] + (cpu_product<<8>>8)  + result_carry);

			//shift carry bits from upper half of short sum
			result_carry = (cpu_sum >> 8);

			//update current indexs result
			cpu_result[result_position] = cpu_sum;

			result_position++;
			loop++;
		}

		//compute final carry of last index from each row
		cpu_sum = (cpu_result[result_position] + result_carry);
		cpu_result[result_position] = cpu_sum;

		//update carry for those rows which are not equal to n
		result_carry = (cpu_sum >> 8);
		cpu_result[result_position+ 1] += result_carry;

}

	gettimeofday(&cpu_end, &tzp); // return is in microseconds
	printf("CPU time: %.6f\n", (cpu_end.tv_sec - cpu_start.tv_sec) + (cpu_end.tv_usec - cpu_start.tv_usec) / 1000000.0);

        unsigned int err = 0;
        unsigned int g = 0;
	
	//compare for finding error in the result of cpu vs. gpu
	while(g<2*n){
                if(final[g] != cpu_result[g]) {
                        err++;
                }
                g++;
        }

        if(err == 0)
                printf("PASS\n");

        else
                printf("FAIL\n");
	
	//free memory
	free(p);
	free(q);
	free(cpu_result);
	free(final);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

return 0;
}

//each thread will compute a complete row of products where the index of the kernel array is the multiplicand
//for the specific threads multiplicand. The thread will loop through the other multilpiers to calculate a
//row of products. Atomically add to assure that data is not missed or overwritten.
__global__ void get_products(unsigned char a[], unsigned char b[], unsigned int accumulator[], unsigned int n) {

	int multiplier = 0;
	unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;// kernel index
	unsigned int multiplicand = index;

	//atomic add only uses integers so product will only store a short becuase atomic add integer
	//cannot be casted to a short
	unsigned int product = 0;

	//loop through multipliers and find products
	while(multiplier < n) {
	
		//compute ch * ch and produce a short
		product = (unsigned int) a[multiplier] * b[multiplicand];

		//add the first character to the respective result index
		atomicAdd(&accumulator[multiplier + index], product<<24>>24);
		
		//add the second character to the respective result index
		atomicAdd(&accumulator[multiplier + index + 1], product>>8);
		multiplier++;
	}
return;
}
