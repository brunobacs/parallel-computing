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
    
    double *A = (double *)malloc(M * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(M * N * sizeof(double));
    
    for (int i = 0; i < M * N; i++) {
        A[i] = i % 100;
        C[i] = 0;
    }
    for (int i = 0; i < N * N; i++) {
        B[i] = (i % 10) + 1;
    }

    // Envia trabalhos pros workers
    int worker_rank = 1;
    for (int px = 0; px < Px; px++) {
        for (int py = 0; py < Py; py++) {
            int start_row = px * block_rows;
            int start_col = py * block_cols;

            printf("Sending block [%d,%d] to worker %d\n", px, py, worker_rank);

            MPI_Send(&block_rows, 1, MPI_INT, worker_rank, ENVIA_TAM, MPI_COMM_WORLD);
            MPI_Send(&block_cols, 1, MPI_INT, worker_rank, ENVIA_TAM, MPI_COMM_WORLD);
            MPI_Send(&N, 1, MPI_INT, worker_rank, ENVIA_TAM, MPI_COMM_WORLD);
        
            MPI_Send(&A[start_row * N], block_rows * N, MPI_DOUBLE, worker_rank, ENVIA_BLOCO, MPI_COMM_WORLD);
            MPI_Send(&B[start_col * N], N * block_cols, MPI_DOUBLE, worker_rank, ENVIA_BLOCO, MPI_COMM_WORLD);
            MPI_Send(&escalar, 1, MPI_DOUBLE, worker_rank, ENVIA_BLOCO, MPI_COMM_WORLD);

            worker_rank++;
        }
    }

    // recebe dos workers
    worker_rank = 1;
    for (int px = 0; px < Px; px++) {
        for (int py = 0; py < Py; py++) {
            int start_row = px * block_rows;
            int start_col = py * block_cols;
            
            double *block_result = (double *)malloc(block_rows * block_cols * sizeof(double));
            MPI_Recv(block_result, block_rows * block_cols, MPI_DOUBLE, worker_rank, RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
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

    printf("\nFinal Result Matrix C:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", C[i * N + j]);
        }
        printf("\n");
    }

    free(A);
    free(B);
    free(C);
}

void slave() {
    MPI_Status status;
    int block_rows, block_cols, N;

    MPI_Recv(&block_rows, 1, MPI_INT, 0, ENVIA_TAM, MPI_COMM_WORLD, &status);
    MPI_Recv(&block_cols, 1, MPI_INT, 0, ENVIA_TAM, MPI_COMM_WORLD, &status);
    MPI_Recv(&N, 1, MPI_INT, 0, ENVIA_TAM, MPI_COMM_WORLD, &status);
    
    double *A_block = (double *)malloc(block_rows * N * sizeof(double));
    double *B_block = (double *)malloc(N * block_cols * sizeof(double));
    double escalar;

    MPI_Recv(A_block, block_rows * N, MPI_DOUBLE, 0, ENVIA_BLOCO, MPI_COMM_WORLD, &status);
    MPI_Recv(B_block, N * block_cols, MPI_DOUBLE, 0, ENVIA_BLOCO, MPI_COMM_WORLD, &status);
    MPI_Recv(&escalar, 1, MPI_DOUBLE, 0, ENVIA_BLOCO, MPI_COMM_WORLD, &status);

    double *C_block = (double *)malloc(block_rows * block_cols * sizeof(double));
    for (int i = 0; i < block_rows; i++) {
        for (int j = 0; j < block_cols; j++) {
            C_block[i * block_cols + j] = 0;
            for (int k = 0; k < N; k++) {
                C_block[i * block_cols + j] += escalar * A_block[i * N + k] * B_block[k * block_cols + j];
            }
        }
    }

    MPI_Send(C_block, block_rows * block_cols, MPI_DOUBLE, 0, RESULT, MPI_COMM_WORLD);

    free(A_block);
    free(B_block);
    free(C_block);
}