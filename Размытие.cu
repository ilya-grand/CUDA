#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<Windows.h>
using namespace std;

#define threads 256
#define h_x 0.1
#define h_y 0.2
#define T_step 5

#define CHECK_ERR(call)\
{\
	cudaError_t err = call;\
	if(err != cudaSuccess)\
	{\
		printf("Ошибка CUDA в файле: <%s>; в строке <%d> - %s\n", __FILE__, __LINE__, cudaGetErrorString);\
		exit(EXIT_FAILURE);\
	}\
}\

#define CHECK_LAST_ERR()\
{\
	cudaError_t err = cudaGetLastError();\
	if(err != cudaSuccess)\
	{\
		printf("Ошибка CUDA  в файле: <%s>; в строке <%d> - %s\n", __FILE__, __LINE__, cudaGetErrorString);\
		exit(EXIT_FAILURE);\
	}\
}\

void show_matrix(double* A, int n, int m)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			printf("%.5f ", A[i * n + j]);
		}
		printf("\n");
	}
}

void CPU_calc(double* A, double* B, int n, int m, double tau)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			if (i == 0 || j == 0 || i == n - 1 || j == m - 1)
			{
				B[i * n + j] = A[i * n + j];
			}
			else
			{
				B[i * n + j] = A[i * n + j] + tau * (((A[(i + 1) * n + j] - 2 * A[i * n + j] + A[(i - 1) * n + j]) / (h_x * h_x)) + ((A[i * n + (j + 1)] - 2 * A[i * n + j] + A[i * n + (j - 1)]) / (h_y * h_y)));
			}
		}
	}
}

int main()
{
	SetConsoleOutputCP(1251);

	double* h_U_P, * h_U_N;
	double* d_U_P, * d_U_N;

	int n = 1 / h_x + 1;
	int m = 1 / h_y + 1;

	double tau = 0.5 / ((1. / (h_x * h_x)) + (1. / (h_y * h_y)));

	int size = m * n;

	double* tmp;

	h_U_P = (double*)malloc(size * sizeof(double));
	h_U_N = (double*)malloc(size * sizeof(double));
	cudaMalloc((void**)&d_U_P, size * sizeof(double));
	cudaMalloc((void**)&d_U_N, size * sizeof(double));

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			if (i == 0 || j == 0 || i == n - 1 || j == m - 1)
			{
				h_U_P[i * n + j] = 2.;
			}
			else
			{
				h_U_P[i * n + j] = 1.;
			}
		}
	}

	show_matrix(h_U_P, n, m);

	cudaMemcpy(d_U_P, h_U_P, size * sizeof(double), cudaMemcpyHostToDevice);

	for (int i = 0; i < T_step; i++)
	{
		CPU_calc(h_U_P, h_U_N, n, m, tau);

		tmp = h_U_P;
		h_U_P = h_U_N;
		h_U_N = tmp;
	}

	printf("\n");
	show_matrix(h_U_P, n, m);

	return 0;
}