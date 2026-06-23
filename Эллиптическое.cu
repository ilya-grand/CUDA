#include<Windows.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<stdlib.h>
#include<ctime>
#include<algorithm>
#include<cmath>
#include<thrust/extrema.h>
#include<thrust/device_ptr.h>
using namespace std;

#define h_x 0.1
#define h_y 0.2
#define EPS 0.1
#define threads 256

void show_matrix(double* A, int w, int h)
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

void CPU_calc(double* A, double* B, int w, int h, double tau)
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
				B[i * w + j] = (((A[(i + 1) * w + j] + A[(i - 1) * w + j]) / (h_x * h_x)) + ((A[i * w + (j + 1)] + A[i * w + (j - 1)]) / (h_y * h_y))) / (((2.) / (h_x * h_x)) + ((2.) / (h_y * h_y)));
			}
		}
	}
}

void CPU_diff(double* A, double* B, double* C, int w, int h)
{
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			C[i * w + j] = abs(B[i * w + j] - A[i * w + j]);
		}
	}
}

__global__ void GPU_calc(double* A, double* B, int w, int h, double tau)
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
			B[i * w + j] = (((A[(i + 1) * w + j] + A[(i - 1) * w + j]) / (h_x * h_x)) + ((A[i * w + (j + 1)] + A[i * w + (j - 1)]) / (h_y * h_y))) / (((2.) / (h_x * h_x)) + ((2.) / (h_y * h_y)));
		}
	}
}

__global__ void GPU_diff(double* A, double* B, double* C, int w, int h)
{
	int lin_ind = blockDim.x * blockIdx.x + threadIdx.x;

	if (lin_ind < w * h)
	{
		int i = lin_ind / w;
		int j = lin_ind % w;

		C[i * w + j] = abs(B[i * w + j] - A[i * w + j]);
	}
}

int main()
{
	SetConsoleOutputCP(1251);

	double* h_U_P, * h_U_N, * h_err_list;
	double* d_U_P, * d_U_N, * d_err_list;

	int w = (int)(1 / h_x) + 1;
	int h = (int)(1 / h_y) + 1;

	int size = w * h;

	int blocks = w * h / threads + 1;

	double tau = 0.5 / ((1. / (h_x * h_x)) + (1. / (h_y * h_y)));

	double* tmp;

	h_U_P = (double*)malloc(size * sizeof(double));
	h_U_N = (double*)malloc(size * sizeof(double));
	h_err_list = (double*)malloc(size * sizeof(double));
	cudaMalloc((void**)&d_U_P, size * sizeof(double));
	cudaMalloc((void**)&d_U_N, size * sizeof(double));
	cudaMalloc((void**)&d_err_list, size * sizeof(double));

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			if (i == 0)
			{
				h_U_P[i * w + j] = exp(1 - h_x * j);
			}
			if (j == 0)
			{
				h_U_P[i * w + j] = exp(1 - h_y * i);
			}
			if (i == h - 1 || j == w - 1)
			{
				h_U_P[i * w + j] = 1;
			}
			if (i > 0 && j > 0 && i < h - 1 && j < w - 1)
			{
				h_U_P[i * w + j] = 0;
			}
		}
	}

	show_matrix(h_U_P, w, h);
	printf("\n");

	cudaMemcpy(d_U_P, h_U_P, size * sizeof(double), cudaMemcpyHostToDevice);

	int c = 0;

	while (true)
	{
		CPU_calc(h_U_P, h_U_N, w, h, tau);
		CPU_diff(h_U_P, h_U_N, h_err_list, w, h);

		show_matrix(h_U_N, w, h);
		printf("\n");

		c++;

		if (*max_element(h_err_list, h_err_list + size) < EPS)
		{
			break;
		}
		
		tmp = h_U_P;
		h_U_P = h_U_N;
		h_U_N = tmp;
	}

	printf("%d\n", c);

	c = 0;

	while (true)
	{
		GPU_calc << <blocks, threads >> > (d_U_P, d_U_N, w, h, tau);
		GPU_diff << <blocks, threads >> > (d_U_P, d_U_N, d_err_list, w, h);

		cudaMemcpy(h_U_N, d_U_N, size * sizeof(double), cudaMemcpyDeviceToHost);

		show_matrix(h_U_N, w, h);
		printf("\n");

		c++;

		thrust::device_ptr<double> dvc_ptr(d_err_list);
		if (*thrust::max_element(dvc_ptr, dvc_ptr + size) < EPS)
		{
			break;
		}

		tmp = d_U_N;
		d_U_N = d_U_P;
		d_U_P = tmp;
	}

	printf("%d\n", c);

	return 0;
}