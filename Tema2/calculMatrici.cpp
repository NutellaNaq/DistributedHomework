// Produsul a doua matrici C = A * B, calculat în paralel cu OpenMP
// pe un calculator SMP (memorie partajata, mai multe thread-uri).
//
// Rulare:             ./calculMatrici [N] [nr_threaduri]      ex: ./calculMatrici 1000 4
#include <omp.h>

#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 1000;   // dimensiunea matricilor N x N
    if (argc > 2) omp_set_num_threads(std::atoi(argv[2]));  // cati threads folosim

    // Matricile sunt stocate liniar: elementul (i, j) se afla la i * n + j
    // Cream matricii cu valori aleatori patratice A, B si initializam C pentru raspuns
    std::vector<double> A(n * n), B(n * n), C(n * n, 0.0);
    for (int i = 0; i < n * n; i++) {
        A[i] = std::rand() % 10;
        B[i] = std::rand() % 10;
    }

    double start = omp_get_wtime();  // Luam timpul 

    // Fiecare thread primeste un grup de linii din C si le calculeaza independent.
    // i, j, k sunt declarate în bucle, deci sunt private fiecarui thread.
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double suma = 0.0;
            for (int k = 0; k < n; k++)
                suma += A[i * n + k] * B[k * n + j];
            C[i * n + j] = suma;
        }

    double timp = omp_get_wtime() - start;   // Scadem timpul actual cu cel initial pentru a afla durata

    std::cout << "N = " << n
              << ", thread-uri = " << omp_get_max_threads()
              << ", timp = " << timp << " s\n";
    std::cout << "C[0][0] = " << C[0] << ", C[n-1][n-1] = " << C[n * n - 1] << "\n";
    return 0;
}
