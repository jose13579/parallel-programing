#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void matriz_soma(int *A, int *B, int *C,int linhas, int colunas, int dim) {
        // Calculate the row index of the C element and A
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    // Calculate the column index of P and B 
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // Calculate the index of position in the block and thread in the X or Y orientation
    int index = row * colunas + col;

    // if row <rows and col <columns additionally the value of the current position
    // in block and thread is smaller than size of rows * columns
    // then it can add the values ​​in two grids A and B
    if((row < linhas && col < colunas) && (index < dim)) {
        
	// Sum A and B
        C[index] = A[index] + B[index];
    }
}


int main()
{
    // Initialize variables
    int *A, *B, *C;
    int i, j;
    int *d_a, *d_b, *d_c;

    int linhas, colunas;

    // Scan values
    scanf("%d", &linhas);
    scanf("%d", &colunas);

    // Calculation of dimensions
    int dim = linhas * colunas;
    int size = dim * sizeof(int);

    // Alloc CPU memory
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Initialize arrays
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            A[i*colunas+j] =  B[i*colunas+j] = i+j;
        }
    }

    // Copy inputs to device
    cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

    // Initialize dimGrid and dimBlocks
    // create a grid with (32 / columns) number of columns and (32 / lines) number of rows
    // the ceiling function makes sure there are enough to cover all elements 
    dim3 dimGrid(ceil((float)colunas / 32), ceil((float)linhas / 32), 1);

    // create a block with 32 columns and 32 rows
    dim3 dimBlock(32, 32, 1);

    // Launch matriz_soma() kernel on GPU with a grid and block as input
    matriz_soma<<<dimGrid, dimBlock>>>(d_a, d_b,d_c,linhas, colunas, dim);

    // Copy result to local array
    cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

    long long int somador=0;

    // Keep this computation in the CPU
    // Sum all values
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            somador+=C[i*colunas+j];
        }
    }

    // print sum
    printf("%lli\n", somador);

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return (0);
}
