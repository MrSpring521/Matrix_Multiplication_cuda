#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#define size 1000

void matex_CPU(float* A,float* B,float* R)
{
    float h,s;
    for(int i=0;i<size;i++)
    {
        for(int q=0;q<size;q++)
        {
            h = 0;
            for(int k=0;k<size;k++)
            {
                s = A[i*size+k]*B[k*size+q];
                h+=s;
            }
            R[i*size+q] = h;
        }
    }
}

__global__ void matex_GPU(float* A,float* B,float* R)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 行号
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列号
    
    if(row < size && col < size) { // 确保不超出矩阵边界
        float value = 0;
        for(int k = 0; k < size; ++k) {
            value += A[row * size + k] * B[k * size + col];
        }
        R[row * size + col] = value;
    }
}

int main(void)
{
    srand((unsigned)time(NULL));
    float *A = (float*)malloc(size * size * sizeof(float));
    float *B = (float*)malloc(size * size * sizeof(float));
    float *RC = (float*)malloc(size * size * sizeof(float));
    float *RG = (float*)malloc(size * size * sizeof(float));
    for(int i=0;i<size;i++)
    {
        for(int j=0;j<size;j++)
        {
            A[i*size+j] = float(rand())/100000;
        }
    }
    for(int i=0;i<size;i++)
    {
        for(int j=0;j<size;j++)
        {
            B[i*size+j] = float(rand())/100000;
        }
    }
    float *RGDevice, *ADevice, *BDevice;
    cudaMalloc((float**)&RGDevice,sizeof(float)*size*size);
    cudaMalloc((float**)&ADevice,sizeof(float)*size*size);
    cudaMalloc((float**)&BDevice,sizeof(float)*size*size);
    cudaMemcpy(RGDevice,RG,sizeof(float)*size*size,cudaMemcpyHostToDevice);
    cudaMemcpy(ADevice,A,sizeof(float)*size*size,cudaMemcpyHostToDevice);
    cudaMemcpy(BDevice,B,sizeof(float)*size*size,cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(32, 32); // 每个block中的线程数,最大为1024
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (size + threadsPerBlock.y - 1) / threadsPerBlock.y); // block的数量
    matex_GPU<<<numBlocks, threadsPerBlock>>>(ADevice,BDevice,RGDevice);
    matex_CPU(A,B,RC);
    cudaDeviceSynchronize();
    cudaMemcpy(RG, RGDevice, sizeof(float)*size*size, cudaMemcpyDeviceToHost);
    int count = 0;
    for(int i=0;i<size;i++)
    {
        for(int j=0;j<size;j++)
        {
            if ((fabs(RG[i*size+j] - RC[i*size+j]))/ RG[i*size+j] > 0.01) 
            {
                count++;
                break;
            }
        }
        if(count!=0)
        {
            break;
        }
    }
    if(count!=0)
    {
        printf("wrong\n");
    }
    else
    {
        printf("right\n");
    }

    // for(int i=0;i<size;i++)
    // {
    //     for(int j=0;j<size;j++)
    //     {
    //         printf("%lf ",RC[i*size+j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // for(int i=0;i<size;i++)
    // {
    //     for(int j=0;j<size;j++)
    //     {
    //         printf("%lf ",RG[i*size+j]);
    //     }
    //     printf("\n");
    // }
    free(A);
    free(B);
    free(RC);
    free(RG);
    cudaFree(RGDevice);
    cudaFree(ADevice);
    cudaFree(BDevice);
    return 0;
}
