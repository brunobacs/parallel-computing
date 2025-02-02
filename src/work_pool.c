#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/work_pool.h"
#include <omp.h>
#include <time.h>

void worker(int rank, int M, int N, int block_rows, int block_cols) {
    MPI_Status status;
    printf("Process %d initialized.\n", rank);

    while (1) {
        MPI_Send(NULL, 0, MPI_BYTE, 0, WORK_REQUEST, MPI_COMM_WORLD);
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TERMINATION) {
            MPI_Recv(NULL, 0, MPI_BYTE, 0, TERMINATION, MPI_COMM_WORLD, &status);
            break;
        } else if (status.MPI_TAG == WORK_ASSIGNMENT) {
            int block_row, block_col;
            double alpha;

            MPI_Recv(&block_row, 1, MPI_INT, 0, WORK_ASSIGNMENT, MPI_COMM_WORLD, &status);
            MPI_Recv(&block_col, 1, MPI_INT, 0, WORK_ASSIGNMENT, MPI_COMM_WORLD, &status);
            MPI_Recv(&alpha, 1, MPI_DOUBLE, 0, WORK_ASSIGNMENT, MPI_COMM_WORLD, &status);

            double *A_block = (double *)malloc(block_rows * N * sizeof(double));
            double *B_block = (double *)malloc(N * block_cols * sizeof(double));

            MPI_Recv(A_block, block_rows * N, MPI_DOUBLE, 0, WORK_ASSIGNMENT, MPI_COMM_WORLD, &status);
            MPI_Recv(B_block, N * block_cols, MPI_DOUBLE, 0, WORK_ASSIGNMENT, MPI_COMM_WORLD, &status);

            double *C_block = (double *)malloc(block_rows * block_cols * sizeof(double));
            for (int i = 0; i < block_rows; i++) {
                for (int j = 0; j < block_cols; j++) {
                    C_block[i * block_cols + j] = 0;
                    for (int k = 0; k < N; k++) {
                        C_block[i * block_cols + j] += alpha * A_block[i * N + k] * B_block[k * block_cols + j];
                    }
                }
            }

            MPI_Send(&block_row, 1, MPI_INT, 0, RESULT, MPI_COMM_WORLD);
            MPI_Send(&block_col, 1, MPI_INT, 0, RESULT, MPI_COMM_WORLD);
            MPI_Send(C_block, block_rows * block_cols, MPI_DOUBLE, 0, RESULT, MPI_COMM_WORLD);

            free(A_block);
            free(B_block);
            free(C_block);
        }
    }
}

void root(int Px, int Py, int M, int N, double escalar, int num_workers) {
    int block_rows = M / Px;
    int block_cols = N / Py;
    int num_blocks = Px * Py;

    printf("Root process initialized with %d workers\n", num_workers);
    printf("Matrix dimensions: %dx%d\n", M, N);
    printf("Block dimensions: %dx%d\n", block_rows, block_cols);
    printf("Number of blocks: %d\n", num_blocks);

    double *A = (double *)malloc(M * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(M * N * sizeof(double));

    srand(time(NULL));

    for (int i = 0; i < M * N; i++) {
        A[i] = rand() % 100;
        C[i] = 0;
    }
    for (int i = 0; i < N * N; i++) {
        B[i] = (rand() % 10) + 1;
    }

    int completed_blocks = 0;
    int current_block = 0;
    MPI_Status status;

    while (completed_blocks < num_blocks) {
        int worker_rank;
        MPI_Recv(NULL, 0, MPI_BYTE, MPI_ANY_SOURCE, WORK_REQUEST, MPI_COMM_WORLD, &status);
        worker_rank = status.MPI_SOURCE;

        if (current_block < num_blocks) {
            int block_row = (current_block / Py) * block_rows;
            int block_col = (current_block % Py) * block_cols;

            printf("Assigning block [%d,%d] to worker %d\n", current_block / Py, current_block % Py, worker_rank);

            MPI_Send(&block_row, 1, MPI_INT, worker_rank, WORK_ASSIGNMENT, MPI_COMM_WORLD);
            MPI_Send(&block_col, 1, MPI_INT, worker_rank, WORK_ASSIGNMENT, MPI_COMM_WORLD);
            MPI_Send(&escalar, 1, MPI_DOUBLE, worker_rank, WORK_ASSIGNMENT, MPI_COMM_WORLD);

            double *A_block = &A[block_row * N];
            double *B_block = &B[block_col * N];

            MPI_Send(A_block, block_rows * N, MPI_DOUBLE, worker_rank, WORK_ASSIGNMENT, MPI_COMM_WORLD);
            MPI_Send(B_block, N * block_cols, MPI_DOUBLE, worker_rank, WORK_ASSIGNMENT, MPI_COMM_WORLD);

            current_block++;
        } else {
            MPI_Send(NULL, 0, MPI_BYTE, worker_rank, TERMINATION, MPI_COMM_WORLD);
        }

        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == RESULT) {
            int block_row, block_col;
            MPI_Recv(&block_row, 1, MPI_INT, status.MPI_SOURCE, RESULT, MPI_COMM_WORLD, &status);
            MPI_Recv(&block_col, 1, MPI_INT, status.MPI_SOURCE, RESULT, MPI_COMM_WORLD, &status);

            double *block_result = (double *)malloc(block_rows * block_cols * sizeof(double));
            MPI_Recv(block_result, block_rows * block_cols, MPI_DOUBLE, status.MPI_SOURCE, RESULT, MPI_COMM_WORLD, &status);

            #pragma omp parallel for
            for (int i = 0; i < block_rows; i++) {
                for (int j = 0; j < block_cols; j++) {
                    C[(block_row + i) * N + (block_col + j)] = block_result[i * block_cols + j];
                }
            }

            free(block_result);
            completed_blocks++;
            printf("Received result from worker %d. Completed blocks: %d/%d\n", status.MPI_SOURCE, completed_blocks, num_blocks);
        }
    }

    printf("\nFinal Result Matrix C:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", C[i * N + j]);
        }
        printf("\n");
    }

    for (int i = 1; i <= num_workers; i++) {
        MPI_Send(NULL, 0, MPI_BYTE, i, TERMINATION, MPI_COMM_WORLD);
    }

    free(A);
    free(B);
    free(C);

}