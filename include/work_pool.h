#ifndef WORKPOOL_H
#define WORKPOOL_H

#define WORK_REQUEST 0
#define WORK_ASSIGNMENT 1
#define RESULT 2
#define TERMINATION 3

void root(int Px, int Py, int M, int N, double escalar, int num_workers);
void worker(int rank, int N, int M);

#endif
