#ifndef MASTER_SLAVE_H
#define MASTER_SLAVE_H

#define ENVIA_TAM 0
#define ENVIA_BLOCO 1
#define RESULT 2
#define TERMINATION 3

void master(int Px, int Py, int M, int N, double escalar);
void slave();
void fill(int n, double* m);


#endif
