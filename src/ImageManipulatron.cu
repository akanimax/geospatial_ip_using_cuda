/*
 ============================================================================
 Name        : ImageManipulatron.cu
 Author      : ANIMESH_K
 Version     :
 Copyright   : all rights reserved
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <fstream>
#include <iosfwd>
#include <iostream>
#include <ostream>
#include <map>
#include <cuda.h>
#define BLOCK_CONSTANT_WIDTH 16
#define NOTHING -99999

using namespace std;


/**
 *
 * This macro function calculates the index into the mem-array
 * for this application only. (The single row-coloumn padding is considered here)
 *
 * ?? Also note that this cannot be a normal function because a host function cannot
 * be called from a device kernel??
 *
 * */
#define CALC_INDEX_2D(pure_x, pure_y, bdim_x, gdim_x)\
	(((gdim_x * bdim_x) + 2) * (pure_y)) + (pure_x)

// note the + 2 is missing
#define CALC_INDEX_2D_UNPADDED(pure_x, pure_y, bdim_x, gdim_x)\
	(((gdim_x * bdim_x)) * (pure_y)) + (pure_x)

/**
 * error checking function:
 * */

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   } else cout << "Success " << endl;
}


__global__ void fdd_on_gpu(int* outputData, float* inputData) {

	// calculate the pure x and y co-ordinates for the current thrread.
	// note that the unit length shift is considered here.
	int pure_x = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
	int pure_y = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

	// create a constant for array-index for difference calculation
	// create a separate array for the difference storage.
	float dif_arr[9]; int arr_ind; // no need to initialize the dif_arr

	// calculate the arr_ind using the above defined function:
	arr_ind = CALC_INDEX_2D(pure_x, pure_y, blockDim.x, gridDim.x);

	// now, calculate all the 8 different differences betweem the center
	// point and surrounding points.

	dif_arr[0] = inputData[arr_ind] - inputData[CALC_INDEX_2D(pure_x - 1, pure_y - 1, blockDim.x, gridDim.x)];
	dif_arr[1] = inputData[arr_ind] - inputData[CALC_INDEX_2D(pure_x, pure_y - 1, blockDim.x, gridDim.x)];
	dif_arr[2] = inputData[arr_ind] - inputData[CALC_INDEX_2D(pure_x + 1, pure_y - 1, blockDim.x, gridDim.x)];
	dif_arr[3] = inputData[arr_ind] - inputData[CALC_INDEX_2D(pure_x - 1, pure_y, blockDim.x, gridDim.x)];
	dif_arr[4] = inputData[arr_ind] - inputData[CALC_INDEX_2D(pure_x, pure_y, blockDim.x, gridDim.x)];
	dif_arr[5] = inputData[arr_ind] - inputData[CALC_INDEX_2D(pure_x + 1, pure_y, blockDim.x, gridDim.x)];
	dif_arr[6] = inputData[arr_ind] - inputData[CALC_INDEX_2D(pure_x - 1, pure_y + 1, blockDim.x, gridDim.x)];
	dif_arr[7] = inputData[arr_ind] - inputData[CALC_INDEX_2D(pure_x, pure_y + 1, blockDim.x, gridDim.x)];
	dif_arr[8] = inputData[arr_ind] - inputData[CALC_INDEX_2D(pure_x + 1, pure_y + 1, blockDim.x, gridDim.x)];

	// find the max difference from this array
	// this is to be performed by every array

	int maxDifference = NOTHING, dir = 4; // means no flow direction initially
	for(int i = 0; i < 8; i++) {
		if(dif_arr[i] > maxDifference) {
			maxDifference = dif_arr[i];
			dir = i;
		}
	}

	// final check on the max_difference
	if(maxDifference > 0) {
		outputData[CALC_INDEX_2D_UNPADDED(pure_x - 1, pure_y - 1, blockDim.x, gridDim.x)] = dir;
	} /* else */
	outputData[CALC_INDEX_2D_UNPADDED(pure_x - 1, pure_y - 1, blockDim.x, gridDim.x)] = 4; // no flow direction
}

int main(int argc, char* argv[]) {

	// check for the validity of the number of arguments
	if(argc != 2) {
		cout << "\n\nUsage: ./ImageManipulatron <ascii file name>" << endl << endl;
		exit(-1);
	}

	/* else */
	// arguments check passed! so, proceed

	ifstream inputFile; // create and open a file stream
	inputFile.open(argv[1]);

	// check to see if the file really exists
	if(!inputFile) {
		cout << "\n\nSorry, Cannot open the file" << endl << endl;
		exit(-2);
	}

	/*
	 * Next part is the code to extract the relevant metadata from the asc file.
	 * */

	map<string, double>  metadata; // create a map for the metadata
	for(int i = 0; i <= 5; i++) {
		string key; double value;
		inputFile >> key; inputFile >> value;
		metadata.insert(pair<string, double>(key, value));
	}

	// iterate over the map and print all the key values in it just to log to the console
	cout << "\nThe extracted metadata from the given file is: " << endl;
	for(map<string, double>::iterator it = metadata.begin();
			it != metadata.end(); it++) {
		cout << it->first << ": " << it->second << endl;
	}

	float nodata_value = metadata["nodata_value"]; // get only the required parameters from the map
	int ncols = metadata["ncols"], nrows = metadata["nrows"];



	// create an array of the size as follows:
	int nrows_padded = nrows + (BLOCK_CONSTANT_WIDTH - (nrows % BLOCK_CONSTANT_WIDTH)) + 2;
	int ncols_padded = ncols + (BLOCK_CONSTANT_WIDTH - (ncols % BLOCK_CONSTANT_WIDTH)) + 2; // pad the data

	float ascii_data[nrows_padded][ncols_padded]; // get all the data into the (host) memory

	// initialize all the memory to 0 values
	for(int i = 0; i < nrows_padded; i++) {
		for(int j = 0; j < ncols_padded; j++) {
			ascii_data[i][j] = 0;
		}
	}


	// loop over the rest of the file and get all the data to
	for(int i = 1; i <= nrows; i++) {
		for(int j = 1; j <= ncols; j++) {
			float data; inputFile >> data;
			if(data != nodata_value) {
				ascii_data[i][j] = data;
			}
		}
	}

	// the data has been kept inside the ascii_data variable.
	// close the file stream since we no longer need it!
	inputFile.close();

	/*
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * The next part of the code is to perform the GPU related steps and then
	 * invoke the defined kernel.
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * */
	float *inputData; int *outputData; // assign pointers for the GPU based memory

	// reserve gpu memory for the two blocks of memories
	cout << "\nAllocating GPU Memory..." << endl;
	cudaMalloc((void**) &inputData, nrows_padded * ncols_padded * sizeof(float));
	cudaMalloc((void**) &outputData, (nrows_padded - 2) * (ncols_padded - 2) * sizeof(int));

	// copy the input data from the host memory to the gpu memory:
	cout << "\nCopying the data to the GPU memory..." << endl;
	cudaMemcpy(inputData, ascii_data, nrows_padded * ncols_padded * sizeof(float), cudaMemcpyHostToDevice);

	// now invoke the kernel such that there are 256 threads per block
	// and every block is a 16 x 16 square.
	dim3 gridDimension((ncols_padded - 2) / BLOCK_CONSTANT_WIDTH,
			(nrows_padded - 2) / BLOCK_CONSTANT_WIDTH);
	dim3 blockDimension(BLOCK_CONSTANT_WIDTH, BLOCK_CONSTANT_WIDTH);

	cout << "\nPerforming Computations..." << endl;
	cout << "BLOCKS INVOKED AS: (" << gridDimension.x << ", " << gridDimension.y << ")" << endl;
	cout << "THREADS INVOKED AS: (" << blockDimension.x << ", " << blockDimension.y << ")" << endl;
	fdd_on_gpu<<< gridDimension, blockDimension >>>(outputData, inputData); // invoke the kernel
	cudaDeviceSynchronize(); // wait for the GPU to exit

	gpuErrchk( cudaPeekAtLastError() ); // check if any error occured


	int flowDirections[nrows_padded - 2][ncols_padded - 2];
	cudaMemcpy(flowDirections, outputData,
			(nrows_padded - 2) * (ncols_padded - 2) * sizeof(int),
			cudaMemcpyDeviceToHost);

	// free the allocated memory on the GPU
	cudaFree(inputData); cudaFree(outputData);

	// to end the program and return successful execution
	return 0;
}


