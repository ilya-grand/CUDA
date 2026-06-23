#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<Windows.h>
#include<ctime>
using namespace std;

#define threads 256
#define h_x 0.001
#define h_y 0.002
#define T_step 50

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

void show_matrix(double* A, int h, int w)
{
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			printf("%.5f ", A[i * w + j]);
		}
		printf("\n");
	}
}

void CPU_calc(double* A, double* B, int h, int w, double tau)
{
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			if (i == 0 || j == 0 || i == h - 1 || j == w - 1)
			{
				B[i * w + j] = A[i * w + j];
			}
			else
			{
				B[i * w + j] = A[i * w + j] + tau * (((A[(i + 1) * w + j] - 2 * A[i * w + j] + A[(i - 1) * w + j]) / (h_x * h_x)) + ((A[i * w + (j + 1)] - 2 * A[i * w + j] + A[i * w + (j - 1)]) / (h_y * h_y)));
			}
		}
	}
}

__global__ void GPU_calc(double* A, double* B, int h, int w, double tau)
{
	int lin_ind = blockDim.x * blockIdx.x + threadIdx.x;

	if (lin_ind < w * h)
	{
		int i = lin_ind / w;
		int j = lin_ind % w;

		if (i == 0 || j == 0 || i == h - 1 || j == w - 1)
		{
			B[i * w + j] = A[i * w + j];
		}
		else
		{
			B[i * w + j] = A[i * w + j] + tau * (((A[(i + 1) * w + j] - 2 * A[i * w + j] + A[(i - 1) * w + j]) / (h_x * h_x)) + ((A[i * w + (j + 1)] - 2 * A[i * w + j] + A[i * w + (j - 1)]) / (h_y * h_y)));
		}
	}
}

int main()
{
	SetConsoleOutputCP(1251);

	double* h_U_P, * h_U_N;
	double* d_U_P, * d_U_N;

	int w = int(1 / h_x) + 1;
	int h = int(1 / h_y) + 1;

	double tau = 0.5 / ((1. / (h_x * h_x)) + (1. / (h_y * h_y)));

	int size = w * h;

	int blocks = w * h / threads + 1;

	double* tmp;

	double CPU_start, CPU_end, CPU_time;
	float GPU_time;
	cudaEvent_t GPU_start, GPU_end;

	h_U_P = (double*)malloc(size * sizeof(double));
	h_U_N = (double*)malloc(size * sizeof(double));
	CHECK_ERR(cudaMalloc((void**)&d_U_P, size * sizeof(double)));
	CHECK_ERR(cudaMalloc((void**)&d_U_N, size * sizeof(double)));

	cudaEventCreate(&GPU_start);
	cudaEventCreate(&GPU_end);

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			if (i == 0 || j == 0 || i == h - 1 || j == w - 1)
			{
				h_U_P[i * w + j] = 2.;
			}
			else
			{
				h_U_P[i * w + j] = 1.;
			}
		}
	}

	//show_matrix(h_U_P, h, w);

	CHECK_ERR(cudaMemcpy(d_U_P, h_U_P, size * sizeof(double), cudaMemcpyHostToDevice));

	CPU_start = clock();
	for (int i = 0; i < T_step; i++)
	{
		CPU_calc(h_U_P, h_U_N, h, w, tau);

		tmp = h_U_P;
		h_U_P = h_U_N;
		h_U_N = tmp;
	}
	CPU_end = clock();
	CPU_time = (double)(CPU_end - CPU_start) / CLOCKS_PER_SEC * 1000;

	printf("Время на CPU: %.5f", CPU_time);

	printf("\n");
	//show_matrix(h_U_P, h, w);

	cudaEventRecord(GPU_start);
	for (int i = 0; i < T_step; i++)
	{
		GPU_calc << <blocks, threads >> > (d_U_P, d_U_N, h, w, tau);
		CHECK_LAST_ERR();

		tmp = d_U_P;
		d_U_P = d_U_N;
		d_U_N = tmp;
	}
	cudaEventRecord(GPU_end);
	cudaEventSynchronize(GPU_end);
	cudaEventElapsedTime(&GPU_time, GPU_start, GPU_end);

	printf("Время на GPU: %.5f", GPU_time);

	CHECK_ERR(cudaMemcpy(h_U_N, d_U_N, size * sizeof(double), cudaMemcpyDeviceToHost));

	printf("\n");
	//show_matrix(h_U_P, h, w);

	return 0;
}