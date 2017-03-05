#include <stdio.h>
#include <time.h>
#include "gputimer.h"

#define ARRAY_ROWS 16000
#define ARRAY_COLS 16000
#define TILE_WIDTH 16

#define cudaCheckError() {                               \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                                \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                      \
    exit(1);                                             \
  }                                                      \
}

void print_matrix( const double * h_A );
void init_matrix( double * h_A );

__global__ void matrix_mul( double *d_A, double *d_C )
{
	// thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	// Each thread stores it's result in this variable
	double tmp  = 0.0;
	
	int row = TILE_WIDTH * blockIdx.y + ty;
	int col = TILE_WIDTH * blockIdx.x + tx;
	
	__shared__ double  Asub[ TILE_WIDTH ][ TILE_WIDTH ];
	__shared__ double  Bsub[ TILE_WIDTH ][ TILE_WIDTH ];

	/* 
	 * We are comparing the blockIdx y & x so we can only calculate
	 * the elements that are on the diagonal & up from them
	 */ 
	if ( blockIdx.y <= blockIdx.x )
	{
	
		for (int i = 0; i < ARRAY_COLS / TILE_WIDTH; i++)
		{
			/**
			*  When the matrix dimensions are not multiples of the tile dimensions,
			*  then it can happen that some tiles cover the matrices only partially
			*  The tile elements falling outside the not-fully overlapping tiles should be properly zero-ed
			*/
			
			if ( i * TILE_WIDTH + tx < ARRAY_ROWS && row < ARRAY_COLS )
				Asub[ ty ][ tx ] = d_A[ row + ARRAY_COLS * ( i * TILE_WIDTH + tx ) ];
			else
				Asub[ ty ][ tx ] = 0.0;

			if ( i * TILE_WIDTH + ty < ARRAY_ROWS && col < ARRAY_COLS )
				Bsub[ ty ][ tx ] = d_A[ col + ARRAY_COLS * ( i * TILE_WIDTH + ty ) ];
			else
				Bsub[ ty ][ tx ] = 0.0;
			
			__syncthreads();
			
			// Calculate only the diagonal and the upper elements
			if ( col >= row )
			{
				tmp += ( Asub[ty][0] * Bsub[0][tx] + Asub[ty][1] * Bsub[1][tx] + Asub[ty][2] * Bsub[2][tx] + Asub[ty][3] * Bsub[3][tx] + Asub[ty][4] * Bsub[4][tx]
				      + Asub[ty][5] * Bsub[5][tx] + Asub[ty][6] * Bsub[6][tx] + Asub[ty][7] * Bsub[7][tx] + Asub[ty][8] * Bsub[8][tx] + Asub[ty][9] * Bsub[9][tx]
				      + Asub[ty][10] * Bsub[10][tx] + Asub[ty][11] * Bsub[11][tx] + Asub[ty][12] * Bsub[12][tx] + Asub[ty][13] * Bsub[13][tx] + Asub[ty][14] * Bsub[14][tx]
				      + Asub[ty][15] * Bsub[15][tx] ); 
			}
			__syncthreads();
			
		}
		
		if ( ( row < ARRAY_COLS ) && ( col < ARRAY_COLS ) && ( row <= col ) )
		{
			d_C[ row * ARRAY_COLS + col ] = tmp; 
			d_C[ col * ARRAY_COLS + row ] = tmp;
		}
	}
}


int main( void )
{
	double ARRAY_BYTES = ARRAY_ROWS * ARRAY_COLS * sizeof( double );
	double C_BYTES    = ARRAY_COLS * ARRAY_COLS * sizeof( double );
	double numOfBlocks;
	GpuTimer timer;
	
	// seed srand function
	srand( time( NULL ) );
	
	// Allocate CPU memory
	double * h_A = ( double * ) malloc( ARRAY_BYTES );
	double * h_C = ( double * ) malloc( C_BYTES );
	
	// Allocate GPU memory
	double * d_A;
	double * d_C;
	
	cudaMalloc( ( void ** ) &d_A, ARRAY_BYTES );
	cudaCheckError();
	cudaMalloc( ( void ** ) &d_C, C_BYTES );
	cudaCheckError();
	
	// Initialize matrix h_A
	init_matrix( h_A );
	
	// Copy Matrix A to GPU
	cudaMemcpy( d_A, h_A, ARRAY_BYTES, cudaMemcpyHostToDevice );
	cudaCheckError();
	
	// Calculate how many blocks need to be created
	dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 );
	
	if ( ARRAY_ROWS >= ARRAY_COLS )
		numOfBlocks = ( double ) ceil( ( double ) ARRAY_ROWS / ( double ) TILE_WIDTH );
	else
		numOfBlocks = ( double ) ceil( ( double ) ARRAY_COLS / ( double ) TILE_WIDTH );
		
	dim3 dimGrid( numOfBlocks, numOfBlocks );
	
	// Print info about grid/block size
	printf( "A size : (%d X %d)\n", ARRAY_ROWS,ARRAY_COLS );
	printf( "GPU will create : %lf blocks\n", numOfBlocks );
	printf( "GPU will create : %d threads per block\n", TILE_WIDTH*TILE_WIDTH );
	
	// Launch Timer & Kernel
	timer.Start();
	matrix_mul<<< dimGrid, dimBlock >>>( d_A, d_C );
	timer.Stop();
	cudaCheckError();
	
	// Time taken for the calculation 
	printf( "Time elapsed = %g ms\n", timer.Elapsed() );
	
	// Copy result from GPU
	cudaMemcpy( h_C, d_C, C_BYTES, cudaMemcpyDeviceToHost );
	cudaCheckError();
	
	// Optional: Only suitable for small arrays
	// print_matrix( h_C );
	
	// Free GPU memory
	cudaFree( d_A );
	cudaFree( d_C );
	
	// Free CPU memory
	free( h_A );
	free( h_C );
	
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
			printf( "%lf\t", h_A[ j + ARRAY_COLS * i ] );
		}
		printf( "\n" );
	}
}