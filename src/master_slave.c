#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "master_slave.h"


void master(int Px, int Py, int M, int N, double escalar) {
    int block_rows = M / Px;
    int block_cols = N / Py;
    printf("Master process initialized\n");
    printf("Matrix dimensions: %dx%d\n", M, N);
    printf("Block dimensions: %dx%d\n", block_rows, block_cols);
    
    // Initialize matrices
    double *A = (double *)malloc(M * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(M * N * sizeof(double));

    // Initialize matrices with values
    for (int i = 0; i < M * N; i++) {
        A[i] = i % 100;
        C[i] = 0;
    }
    for (int i = 0; i < N * N; i++) {
        B[i] = (i % 10) + 1;
    }

    // Distribute blocks to workers
    int worker_rank = 1;
    for (int px = 0; px < Px; px++) {
        for (int py = 0; py < Py; py++) {
            // Calculate block indices
            int start_row = px * block_rows;
            int start_col = py * block_cols;

            printf("Sending block [%d,%d] to worker %d\n", px, py, worker_rank);

            // envia tam blocos e tam da linha
            MPI_Send(&block_rows, 1, MPI_INT, worker_rank, ENVIA_TAM, MPI_COMM_WORLD);
            MPI_Send(&block_cols, 1, MPI_INT, worker_rank, ENVIA_TAM, MPI_COMM_WORLD);
            MPI_Send(&N, 1, MPI_INT, worker_rank, ENVIA_TAM, MPI_COMM_WORLD);
            
            // envia blocos da matriz
            MPI_Send(&A[start_row * N], block_rows * N, MPI_DOUBLE, worker_rank, ENVIA_BLOCO, MPI_COMM_WORLD);
            MPI_Send(&B[start_col], N * block_cols, MPI_DOUBLE, worker_rank, ENVIA_BLOCO, MPI_COMM_WORLD);
            MPI_Send(&escalar, 1, MPI_DOUBLE, worker_rank, ENVIA_BLOCO, MPI_COMM_WORLD);

            worker_rank++;
        }
    }

    // Recebe C calculado
    worker_rank = 1;
    for (int px = 0; px < Px; px++) {
        for (int py = 0; py < Py; py++) {
            int start_row = px * block_rows;
            int start_col = py * block_cols;
            
            // Receive block result
            double *block_result = (double *)malloc(block_rows * block_cols * sizeof(double));
            MPI_Recv(block_result, block_rows * block_cols, MPI_DOUBLE, worker_rank, RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Store result in matrix C
            for (int i = 0; i < block_rows; i++) {
                for (int j = 0; j < block_cols; j++) {
                    C[(start_row + i) * N + (start_col + j)] = block_result[i * block_cols + j];
                }
            }
            
            free(block_result);
            printf("Received result from worker %d\n", worker_rank);
            worker_rank++;
        }
    }

    // Print result matrix
    printf("\nFinal Result Matrix C:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", C[i * N + j]);
        }
        printf("\n");
    }

    // Free memory
    free(A);
    free(B);
    free(C);
}

void slave() {
    MPI_Status status;
    int block_rows, block_cols, N;

    // Receive block dimensions and N
    MPI_Recv(&block_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&block_cols, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    
    // Allocate memory for blocks with correct sizes
    double *A_block = (double *)malloc(block_rows * N * sizeof(double));
    double *B_block = (double *)malloc(N * block_cols * sizeof(double));
    double escalar;

    // Receive data from master
    MPI_Recv(A_block, block_rows * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(B_block, N * block_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&escalar, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);

    // Calculate block of C
    double *C_block = (double *)malloc(block_rows * block_cols * sizeof(double));
    for (int i = 0; i < block_rows; i++) {
        for (int j = 0; j < block_cols; j++) {
            C_block[i * block_cols + j] = 0;
            for (int k = 0; k < N; k++) {
                C_block[i * block_cols + j] += escalar * A_block[i * N + k] * B_block[k * block_cols + j];
            }
        }
    }

    // Send result back to master
    MPI_Send(C_block, block_rows * block_cols, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

    // Free memory
    free(A_block);
    free(B_block);
    free(C_block);
}