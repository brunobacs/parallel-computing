#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/work_pool.h"


void worker(int rank, int M, int N) {
    MPI_Status status;
    printf("Process %d initialized.\n", rank);
    
    while (1) {
        // Request work from master
        MPI_Send(NULL, 0, MPI_BYTE, 0, WORK_REQUEST, MPI_COMM_WORLD);
        
        // Probe for message from master
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
        if (status.MPI_TAG == TERMINATION) {
            MPI_Recv(NULL, 0, MPI_BYTE, 0, TERMINATION, MPI_COMM_WORLD, &status);
            break;
        } else if (status.MPI_TAG == WORK_ASSIGNMENT) {
            int block_row, block_col;
            double alpha;
            
            // Receive block coordinates and scalar
            MPI_Recv(&block_row, 1, MPI_INT, 0, WORK_ASSIGNMENT, MPI_COMM_WORLD, &status);
            MPI_Recv(&block_col, 1, MPI_INT, 0, WORK_ASSIGNMENT, MPI_COMM_WORLD, &status);
            MPI_Recv(&alpha, 1, MPI_DOUBLE, 0, WORK_ASSIGNMENT, MPI_COMM_WORLD, &status);
            
            // Get the count of elements to receive
            int block_rows = 2;  // M/2 for 2x2 blocks
            int block_cols = 2;  // N/2 for 2x2 blocks
            
            // Allocate memory for blocks
            double *A_block = (double *)malloc(block_rows * N * sizeof(double));
            double *B_block = (double *)malloc(N * block_cols * sizeof(double));
            
            // Receive block data
            MPI_Recv(A_block, block_rows * N, MPI_DOUBLE, 0, WORK_ASSIGNMENT, MPI_COMM_WORLD, &status);
            MPI_Recv(B_block, N * block_cols, MPI_DOUBLE, 0, WORK_ASSIGNMENT, MPI_COMM_WORLD, &status);
            
            // Calculate block result
            double *C_block = (double *)malloc(block_rows * block_cols * sizeof(double));
            
            // Proper matrix multiplication for the block
            for (int i = 0; i < block_rows; i++) {
                for (int j = 0; j < block_cols; j++) {
                    C_block[i * block_cols + j] = 0;
                    for (int k = 0; k < N; k++) {
                        C_block[i * block_cols + j] += alpha * A_block[i * N + k] * B_block[k * block_cols + j];
                    }
                }
            }
            
            // Send results back to master
            MPI_Send(&block_row, 1, MPI_INT, 0, RESULT, MPI_COMM_WORLD);
            MPI_Send(&block_col, 1, MPI_INT, 0, RESULT, MPI_COMM_WORLD);
            MPI_Send(C_block, block_rows * block_cols, MPI_DOUBLE, 0, RESULT, MPI_COMM_WORLD);
            
            // Free allocated memory
            free(A_block);
            free(B_block);
            free(C_block);
        }
    }
}

void root(int Px, int Py, int M, int N, double escalar, int num_workers) {
    int block_rows = M / Px;  // Should be 2 for 4x4 matrix with Px=2
    int block_cols = N / Py;  // Should be 2 for 4x4 matrix with Py=2
    int num_blocks = Px * Py; // Should be 4 for 2x2 decomposition

    printf("Root process initialized with %d workers\n", num_workers);
    printf("Matrix dimensions: %dx%d\n", M, N);
    printf("Block dimensions: %dx%d\n", block_rows, block_cols);
    printf("Number of blocks: %d\n", num_blocks);

    // Initialize matrices A, B and C
    double *A = (double *)malloc(M * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(M * N * sizeof(double));
    
    // Initialize matrices with sample values
    for (int i = 0; i < M * N; i++) {
        A[i] = i % 100;
        C[i] = 0;
    }
    for (int i = 0; i < N * N; i++) {
        B[i] = (i % 10) + 1;
    }

    int completed_blocks = 0;
    int current_block = 0;
    MPI_Status status;

    // Process work requests and results
    while (completed_blocks < num_blocks) {
        int worker_rank;
        // Receive message from any worker
        MPI_Recv(NULL, 0, MPI_BYTE, MPI_ANY_SOURCE, WORK_REQUEST, MPI_COMM_WORLD, &status);
        worker_rank = status.MPI_SOURCE;
        
        if (current_block < num_blocks) {
            // Calculate block indices
            int block_row = (current_block / Py) * block_rows;
            int block_col = (current_block % Py) * block_cols;

            printf("Assigning block [%d,%d] to worker %d\n", 
                    current_block / Py, current_block % Py, worker_rank);

            // Send work assignment
            MPI_Send(&block_row, 1, MPI_INT, worker_rank, WORK_ASSIGNMENT, MPI_COMM_WORLD);
            MPI_Send(&block_col, 1, MPI_INT, worker_rank, WORK_ASSIGNMENT, MPI_COMM_WORLD);
            MPI_Send(&escalar, 1, MPI_DOUBLE, worker_rank, WORK_ASSIGNMENT, MPI_COMM_WORLD);

            // Send matrix blocks
            double *A_block = &A[block_row * N];
            double *B_block = &B[block_col];
            
            MPI_Send(A_block, block_rows * N, MPI_DOUBLE, worker_rank, WORK_ASSIGNMENT, MPI_COMM_WORLD);
            MPI_Send(B_block, N * block_cols, MPI_DOUBLE, worker_rank, WORK_ASSIGNMENT, MPI_COMM_WORLD);

            current_block++;
        } else {
            // No more work, send termination signal
            MPI_Send(NULL, 0, MPI_BYTE, worker_rank, TERMINATION, MPI_COMM_WORLD);
        }

        // Check for results
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == RESULT) {
            int block_row, block_col;
            
            // Receive block coordinates
            MPI_Recv(&block_row, 1, MPI_INT, status.MPI_SOURCE, RESULT, MPI_COMM_WORLD, &status);
            MPI_Recv(&block_col, 1, MPI_INT, status.MPI_SOURCE, RESULT, MPI_COMM_WORLD, &status);

            // Receive computed block
            double *block_result = (double *)malloc(block_rows * block_cols * sizeof(double));
            MPI_Recv(block_result, block_rows * block_cols, MPI_DOUBLE, status.MPI_SOURCE, RESULT, 
                    MPI_COMM_WORLD, &status);

            // Store result in matrix C
            for (int i = 0; i < block_rows; i++) {
                for (int j = 0; j < block_cols; j++) {
                    C[(block_row + i) * N + (block_col + j)] = block_result[i * block_cols + j];
                }
            }

            free(block_result);
            completed_blocks++;
            printf("Received result from worker %d. Completed blocks: %d/%d\n", 
                   status.MPI_SOURCE, completed_blocks, num_blocks);
        }
    }

    // Print final result matrix
    printf("\nFinal Result Matrix C:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", C[i * N + j]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(A);
    free(B);
    free(C);
}