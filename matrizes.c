#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/work_pool.h"
#include "include/master_slave.h"

int main(int argc, char *argv[]) {
    if (argc < 7) {
        printf("Use: mpirun -np <n_processos> ./programa <'slave' ou 'pool'> linhas colunas Px Py escalar\n");
        return 1;
    }

    int rank, size, linhas, colunas, escalar, estrategia, Px, Py;

    if (strcmp(argv[1], "slave") == 0) {
        estrategia = 1;
    } else if (strcmp(argv[1], "pool") == 0) {
        estrategia = 2;
    } else {
        printf("Erro: estratégia inválida. Use 'slave' ou 'pool'.\n");
        return 1;
    }

    char *endptr;
    linhas = strtol(argv[2], &endptr, 10);
    if (*endptr != '\0') {
        printf("Erro: o argumento 'linhas' deve ser um número inteiro.\n");
        return 1;
    }

    colunas = strtol(argv[3], &endptr, 10);
    if (*endptr != '\0') {
        printf("Erro: o argumento 'colunas' deve ser um número inteiro.\n");
        return 1;
    }

    Px = strtol(argv[4], &endptr, 10);
    if (*endptr != '\0') {
        printf("Erro: o argumento 'Px' deve ser um número inteiro.\n");
        return 1;
    }

    Py = strtol(argv[5], &endptr, 10);
    if (*endptr != '\0') {
        printf("Erro: o argumento 'Py' deve ser um número inteiro.\n");
        return 1;
    }

    escalar = strtol(argv[6], &endptr, 10);
    if (*endptr != '\0') {
        printf("Erro: o argumento 'escalar' deve ser um número inteiro.\n");
        return 1;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (Px == 0 || Py == 0) {
        if (rank == 0) {
            printf("Erro: Px e Py não podem ser zero.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (linhas % Px != 0 || colunas % Py != 0) {
        if (rank == 0) {
            printf("Erro: Px (%d) deve dividir linhas (%d) e Py (%d) deve dividir colunas (%d).\n", Px, linhas, Py, colunas);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (estrategia == 1) {
        int expected_workers = Px * Py + 1; // +1 para o mestre
        if (size != expected_workers) {
            printf("Erro: o número de processos deve ser Px * Py + 1 (%d).\n", expected_workers);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (rank == 0) {
            master(Px, Py, linhas, colunas, escalar);
        } else {
            slave();
        }
    } else if (estrategia == 2) {
        if (rank == 0) {
            root(Px, Py, linhas, colunas, escalar, (size - 1));
        } else {
            worker(rank, linhas, colunas, linhas / Px, colunas / Py);
        }
    }

    MPI_Finalize();
    return 0;
}