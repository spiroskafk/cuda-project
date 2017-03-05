#include "gputimer.h"
#include <stdio.h>
#include <cublas_v2.h>

#define ARRAY_ROWS 13000
#define ARRAY_COLS 14000

#define cudaCheckError() {                               \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                                \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                      \
    exit(1);                                             \
  }                                                      \
}

// Function prototypes
void init_matrix( double * h_A );
void print_matrix( const double *A );

void gpu_matrix_mul( const double *A, double *C, const int m, const int n )
{
	const double alf     = 1;
	const double bet     = 0;
	const double *alpha  = &alf;
	const double *beta   = &bet;

	// Create a handle for cublas
	cublasHandle_t handle;
	cublasCreate( &handle );
	cudaCheckError();

	// Perform multiplication with cublas
	if ( ARRAY_COLS == ARRAY_ROWS )
		cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_T, m, m, n, alpha, A, m, A, m, beta, C, m  );
	else
		cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, m, alpha, A, n, A, n, beta, C, n  );
		
	cudaCheckError();	
	
	cublasDestroy( handle );
	cudaCheckError();
}


int main( void )
{
	const double ARRAY_BYTES = ARRAY_ROWS * ARRAY_COLS * sizeof( double );
	const double C_BYTES 	  = ARRAY_COLS * ARRAY_COLS * sizeof( double );
	GpuTimer timer;
	
	// seed srand function
	srand( time( NULL ) );

	// Allocate arrays on CPU
	double * h_A = ( double * ) malloc( ARRAY_BYTES );
	double * h_C = ( double * ) malloc( C_BYTES );
	
	// Allocate arrays on GPU
	double * d_A;
	double * d_C;
	cudaMalloc( ( void ** ) &d_A, ARRAY_BYTES );
	cudaCheckError();
	cudaMalloc( ( void ** ) &d_C, C_BYTES );
	cudaCheckError();
	
	// Initialize matrix A with random double numbers
	init_matrix( h_A );
	
	// Copy Matrix A to GPU
	cudaMemcpy( d_A, h_A, ARRAY_BYTES, cudaMemcpyHostToDevice );
	cudaCheckError();
	
	// Launch Timer & Kernel 
	timer.Start();
	gpu_matrix_mul( d_A, d_C, ARRAY_ROWS, ARRAY_COLS );
	timer.Stop();
	
	// Get result from GPU
	cudaMemcpy( h_C, d_C, C_BYTES, cudaMemcpyDeviceToHost );
	cudaCheckError();
	
	// Optional: Print matrix 
	// print_matrix( h_C );
	
	// Time taken for the calculation 
	printf( "Time elapsed = %g ms\n", timer.Elapsed() );
	
	// Free GPU memory
	cudaFree( d_A );
	cudaCheckError();
	cudaFree( d_C );
	cudaCheckError();
	
	// Free CPU memory
	free( h_A );
	free( h_C );

	return 0;
}

void init_matrix( double * h_A )
{
	// Populate array A with random numbers
	for ( int i = 0; i < ARRAY_ROWS; i++ )
		for ( int j = 0; j < ARRAY_COLS; j++ )
			h_A[j * ARRAY_ROWS + i ] = ( double ) rand() / ( double ) ( RAND_MAX );
}

void print_matrix( const double *A )
{
	for ( int i = 0; i < ARRAY_COLS; i++ )
	{
		for ( int j = 0; j < ARRAY_COLS; j++ )
		{
			printf( "%lf\t", A[j * ARRAY_COLS + i] );
		}
		printf( "\n" );
	}
}






