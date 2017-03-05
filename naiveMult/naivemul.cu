#include <stdio.h>
#include <time.h>
#include "gputimer.h"

#define ARRAY_ROWS 16000
#define ARRAY_COLS 16000
#define TILE_WIDTH 16

#define cudaCheckError() {                             \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                               \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
            cudaGetErrorString(e));                       \
    exit(1);                                             \
  }                                                      \
}

// Function protorypes
void print_matrix( const double * h_A );
void init_matrix( double * h_A );


__global__ void matrix_mul( double *d_A, double *d_C, int width )
{
	// Each thread stores it's result in this variable
	double tmp = 0.0;
	
	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x; 
	
	// Out of bounds threads should do nothing
	if ( ( row < width ) && ( col < width ) )
	{
		for ( int k = 0; k < width; k++ )
		{
			tmp += d_A[ k * width + row ] * d_A[ k * width + col ];
		}
		d_C[ row * width + col ] = tmp;
	}	
}

int main( void )
{
	double ARRAY_BYTES = ARRAY_ROWS * ARRAY_COLS * sizeof( double );
	double C_BYTES    = ARRAY_COLS * ARRAY_COLS * sizeof( double );
	GpuTimer timer;
	
	// seed srand function
	srand( time( NULL ) );
	
	// Αllocate CPU memory
	double * h_A = ( double * ) malloc( ARRAY_BYTES );
	double * h_C = ( double * ) malloc( C_BYTES );
	
	// Allocate GPU memory
	double * d_A;
	double * d_C;
	
	cudaMalloc( ( void ** ) &d_A, ARRAY_BYTES );
	cudaCheckError();
	cudaMalloc( ( void ** ) &d_C, C_BYTES );
	cudaCheckError();
	
	// initialize matrix h_A
	init_matrix( h_A );
	
	// Copy matrix A to GPU
	cudaMemcpy( d_A, h_A, ARRAY_BYTES, cudaMemcpyHostToDevice );
	cudaCheckError();
	
	// Calculate how many blocks need to be created
	dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 );
	double numOfBlocks = ( double ) ceil( ( double ) ARRAY_COLS / ( double ) TILE_WIDTH );
	dim3 dimGrid( numOfBlocks, numOfBlocks );
	
	// Print info about grid/block size
	printf( "Array size : ( %d X %d )\n", ARRAY_ROWS, ARRAY_COLS );
	printf( "GPU will create : %lf blocks\n", numOfBlocks );
	printf( "GPU will create : %d threads per block\n", TILE_WIDTH * TILE_WIDTH );

	// Launch Timer & Kernel
	timer.Start();
	matrix_mul<<< dimGrid, dimBlock >>>( d_A, d_C, ARRAY_COLS );
	timer.Stop();
	cudaCheckError();
	
	// Time taken for the calculation 
	printf( "Time elapsed = %g ms\n", timer.Elapsed() );
	
	// Copy result from GPU
	cudaMemcpy( h_C, d_C, C_BYTES, cudaMemcpyDeviceToHost );
	cudaCheckError();
	
	// Optional: Only suitable for small arrays
	// print_matrix( h_C );
	
	// Free CPU memory
	free( h_A );
	free( h_C );
	
	// Free GPU memory
	cudaFree( d_A );
	cudaFree( d_C );
	
	return 0;
}

void init_matrix( double * h_A )
{
	// Initialize array A with random double numbers
	for ( int i = 0; i < ARRAY_ROWS; i++ )
		for ( int j = 0; j < ARRAY_COLS; j++ )
			h_A[j * ARRAY_ROWS + i ] = ( double ) rand() / ( double ) ( RAND_MAX );
}

void print_matrix( const double * h_A )
{
	for ( int i = 0; i < ARRAY_COLS; i++ )
	{
		for ( int j = 0; j < ARRAY_COLS; j++ )
		{
			printf( "%lf\t", h_A[ j * ARRAY_COLS + i ] );
		}
		printf( "\n" );
	}
}