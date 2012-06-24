#include <omp.h>//http://www.microsoft.com/download/en/details.aspx?displaylang=en&id=11310
#include <time.h>
#include <iostream>
#include <conio.h>
#include <windows.h>
#include <intrin.h>
#include <stdlib.h>
#include <math.h>
#include "2DArray.h"
#include <cuda.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
using namespace std;
// Thread block size 
#define BLOCK_SIZE 16 
// Forward declaration of the device multiplication function 
__global__ void Muld(double*, double*, int, int, double*); 
// Host multiplication function 
// Compute C = A * B 
//   hA is the height of A 
//   wA is the width of A 
//   wB is the width of B 
void Mul(const double* A, const double* B, int hA, int wA, int wB, 
         double* C) 
{ 
    int size; 
    // Load A and B to the device 
    double* Ad; 
    size = hA * wA * sizeof(double); 
    cudaMalloc((void**)&Ad, size); 
    cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice); 
    double* Bd; 
    size = wA * wB * sizeof(double); 
    cudaMalloc((void**)&Bd, size); 
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice); 
    // Allocate C on the device 
    double* Cd; 
    size = hA * wB * sizeof(double); 
    cudaMalloc((void**)&Cd, size); 
    // Compute the execution configuration assuming 
    // the matrix dimensions are multiples of BLOCK_SIZE 
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 dimGrid(wB / dimBlock.x, hA / dimBlock.y); 
    // Launch the device computation 
    Muld<<<dimGrid, dimBlock>>>(Ad, Bd, wA, wB, Cd); 
    // Read C from the device 
    cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);  
    // Free device memory 
    cudaFree(Ad); 
    cudaFree(Bd); 
    cudaFree(Cd); 
} 
// Device multiplication function called by Mul() 
// Compute C = A * B 
//   wA is the width of A 
//   wB is the width of B 
__global__ void Muld(double* A, double* B, int wA, int wB, double* C) 
{ 
    // Block index 
    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    // Thread index 
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    // Index of the first sub-matrix of A processed by the block 
    int aBegin = wA * BLOCK_SIZE * by; 
    // Index of the last sub-matrix of A processed by the block 
    int aEnd   = aBegin + wA - 1; 
    // Step size used to iterate through the sub-matrices of A 
    int aStep  = BLOCK_SIZE; 
    // Index of the first sub-matrix of B processed by the block 
    int bBegin = BLOCK_SIZE * bx; 
    // Step size used to iterate through the sub-matrices of B 
    int bStep  = BLOCK_SIZE * wB; 
    // The element of the block sub-matrix that is computed 
    // by the thread 
    double Csub = 0; 
    // Loop over all the sub-matrices of A and B required to 
    // compute the block sub-matrix 
    for (int a = aBegin, b = bBegin; 
             a <= aEnd; 
             a += aStep, b += bStep) { 
        // Shared memory for the sub-matrix of A 
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE]; 
        // Shared memory for the sub-matrix of B 
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE]; 
        // Load the matrices from global memory to shared memory; 
        // each thread loads one element of each matrix 
        As[ty][tx] = A[a + wA * ty + tx]; 
        Bs[ty][tx] = B[b + wB * ty + tx]; 
        // Synchronize to make sure the matrices are loaded 
        __syncthreads(); 
        // Multiply the two matrices together; 
        // each thread computes one element 
        // of the block sub-matrix 
        for (int k = 0; k < BLOCK_SIZE; ++k)	
			Csub += As[ty][k] * Bs[k][tx]; 
        // Synchronize to make sure that the preceding 
        // computation is done before loading two new 
        // sub-matrices of A and B in the next iteration 
        __syncthreads(); 
    } 
    // Write the block sub-matrix to global memory; 
    // each thread writes one element 
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx; 
    C[c + wB * ty + tx] = Csub; 
} 
int main(int argc, char* argv[])  
{      
  

  int M = 1000;
  int N = 1000;
  

  double **A = Allocate2DArray< double >(M, N);
  double **B = Allocate2DArray< double >(M, N);
  double **C = Allocate2DArray< double >(M, N);
  //double **C4 = Allocate2DArray< double >(M, N);
  int i, j;   

  for (i = 0; i < M; i++) {   
    for (j = 0; j < N; j++) {   
		A[i][j] = 5.0 - ((double)(rand()%100) / 10.0);  
    }      
  }   

  for (i = 0; i < M; i++) {   
    for (j = 0; j < N; j++) {   
      B[i][j] = 5.0 - ((double)(rand()%100) / 10.0);   
    }      
  } 
  double start=omp_get_wtime();
 // #pragma omp parallel
  Mul(*A,*B,M,N,N,*C);
  double end=omp_get_wtime();
  cout << "Time is (FOR PARALLEL) " << end-start << " seconds." << endl;
  system("PAUSE");
}