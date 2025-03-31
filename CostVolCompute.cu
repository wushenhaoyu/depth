#include "CostVolCompute.h"
#include "DataParameter.cuh"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>  // 提供CUDA图像处理函数（如cvtColor）
#include <opencv2/cudafilters.hpp>   // 提供CUDA滤波器（如Sobel）
#include <opencv2/cudaarithm.hpp>
#include <iostream>

using namespace cv;
using namespace std;

CostVolCompute::CostVolCompute() {
    // 构造函数实现
}

CostVolCompute::~CostVolCompute() {
    // 析构函数实现
}

// 自定义结构体
struct CudaPoint2f {
    float x, y;

    // 默认构造函数
    __device__ __host__ CudaPoint2f() : x(0), y(0) {}

    // 带参数的构造函数
    __device__ __host__ CudaPoint2f(float x, float y) : x(x), y(y) {}
};

// CUDA 内核：RGB 转灰度图
__global__ void rgbToGrayKernel(int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float r = d_inputImg[idx * channels + 0];
    float g = d_inputImg[idx * channels + 1];
    float b = d_inputImg[idx * channels + 2];
    d_grayImg[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
}

// CUDA 内核：计算梯度图
__global__ void computeGradientKernel(int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float gradX = 0.0f, gradY = 0.0f;

    if (x > 0 && x < width - 1) {
        gradX = d_grayImg[idx + 1] - d_grayImg[idx - 1];
    }
    if (y > 0 && y < height - 1) {
        gradY = d_grayImg[idx + width] - d_grayImg[idx - width];
    }

    d_gradImg[idx] = sqrtf(gradX * gradX + gradY * gradY);
}

// CUDA 内核：计算成本体积
__global__ void computeCostVolKernel(
    int width, int height, int channels,
    int NEIGHBOR_MATCH_LENS_NUM_) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int curCenterIndex = (y / d_microImageParameter.m_circleDiameter) * d_rawImageParameter.m_xLensNum + (x / d_microImageParameter.m_circleDiameter);
    int py = y - d_rawImageParameter.m_yPixelBeginOffset;
    int px = x - d_rawImageParameter.m_xPixelBeginOffset;

    if (d_microImageParameter.m_ppPixelsMappingSet[py * width + px] != curCenterIndex) return;

    const Point2d centerPos = d_microImageParameter.m_ppLensCenterPoints[curCenterIndex];
    const MatchNeighborLens* matchNeighborLens = &d_microImageParameter.m_ppMatchNeighborLens[curCenterIndex * NEIGHBOR_MATCH_LENS_NUM_];

    for (int d = 0; d < d_disparityParameter.m_disNum ; d++) {
        float realDisp = d_disparityParameter.m_dispStep * d + d_disparityParameter.m_dispMin;
        float tempSumCost = 0.0f;
        int tempCostNum = 0;

        for (int i = 0; i < NEIGHBOR_MATCH_LENS_NUM_; i++) {
            float matchCenterPos_y = matchNeighborLens[i].m_centerPosY;
            float matchCenterPos_x = matchNeighborLens[i].m_centerPosX;
            float centerDis = matchNeighborLens[i].m_centerDis;

            if (matchCenterPos_y < 0) break;

            float2 matchPoint;
            matchPoint.y = (centerDis + realDisp) * (matchCenterPos_y - centerPos.y) / centerDis + py;
            matchPoint.x = (centerDis + realDisp) * (matchCenterPos_x - centerPos.x) / centerDis + px;

            if (matchPoint.y < 0 || matchPoint.y >= height || matchPoint.x < 0 || matchPoint.x >= width) continue;

            int matchCenterIndex = matchNeighborLens[i].m_centerIndex;
            if (d_microImageParameter.m_ppPixelsMappingSet[int(matchPoint.y) * width + int(matchPoint.x)] != matchCenterIndex) continue;
            
            float cost = 0.0f;
            if (int(matchPoint.y) == matchPoint.y && int(matchPoint.x) == matchPoint.x) {
                
                const float* curRGB = &d_inputImg[(y * width + x) * channels];
                const float curGrad = d_gradImg[y * width + x];
                const float* matchRGB = &d_inputImg[(int(matchPoint.y) * width + int(matchPoint.x)) * channels];
                const float matchGrad = d_gradImg[int(matchPoint.y) * width + int(matchPoint.x)];

                float clrDiff = 0.0f;
                for (int c = 0; c < channels; c++) {
                    clrDiff += fabs(curRGB[c] - matchRGB[c]);
                }
                clrDiff *= (1.0f / channels);
                float grdDiff = fabs(curGrad - matchGrad);
                cost = grdDiff;
            } else {
                int tempRx = int(matchPoint.x), tempRy = int(matchPoint.y);
                double alphaX = matchPoint.x - tempRx, alphaY = matchPoint.y - tempRy;

                float tempRc[3];
                const float* rgb_y1 = &d_inputImg[(tempRy * width + tempRx) * channels];
                const float* rgb_y2 = &d_inputImg[((tempRy + 1) * width + tempRx) * channels];

                for (int i = 0; i < channels; i++) {
                    tempRc[i] = (1 - alphaX) * (1 - alphaY) * rgb_y1[i] +
                                alphaX * (1 - alphaY) * rgb_y1[i + channels] +
                                (1 - alphaX) * alphaY * rgb_y2[i] +
                                alphaX * alphaY * rgb_y2[i + channels];
                }

                const float* grd_y1 = &d_gradImg[tempRy * width + tempRx];
                const float* grd_y2 = &d_gradImg[(tempRy + 1) * width + tempRx];
                float tempRg = (1 - alphaX) * (1 - alphaY) * grd_y1[0] +
                               alphaX * (1 - alphaY) * grd_y1[1] +
                               (1 - alphaX) * alphaY * grd_y2[0] +
                               alphaX * alphaY * grd_y2[1];

                const float* curRGB = &d_inputImg[(y * width + x) * channels];
                const float curGrad = d_gradImg[y * width + x];

                float clrDiff = 0.0f;
                for (int c = 0; c < channels; c++) {
                    clrDiff += fabs(curRGB[c] - tempRc[c]);
                }
                clrDiff *= (1.0f / channels);
                float grdDiff = fabs(curGrad - tempRg);
                cost = grdDiff;
            }

            tempSumCost += cost;
            tempCostNum++;
        }

        if (tempCostNum > 0) {
            tempSumCost /= tempCostNum;
        }
        
        int costVolIndex = (py * width + px) * d_disparityParameter.m_disNum + d;
        if (costVolIndex >= d_disparityParameter.m_disNum * d_rawImageParameter.m_recImgHeight * d_rawImageParameter.m_recImgWidth) {
            //printf("x:%d y:%d d:%d costVolIndex:%d\n", x, y, d, costVolIndex);
            return;
        }
        d_costVol[costVolIndex] = tempSumCost;
    }
}

// 主函数：计算成本体积
void CostVolCompute::costVolDataCompute(const DataParameter &dataParameter, cv::Mat *costVol) {
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();

    int width = rawImageParameter.m_srcImgWidth;
    int height = rawImageParameter.m_srcImgHeight;
    int channels = 3; // 假设输入图像为 RGB
    int disparityNum = disparityParameter.m_disNum;



    // 将输入图像数据从主机内存复制到设备内存

    // 配置线程块和网格大小
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 启动 CUDA 内核：RGB 转灰度图
    rgbToGrayKernel<<<gridSize, blockSize>>>(width, height, channels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 启动 CUDA 内核：计算梯度图
    computeGradientKernel<<<gridSize, blockSize>>>(width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 启动 CUDA 内核：计算成本体积
    computeCostVolKernel<<<gridSize, blockSize>>>(
        width, height, channels,
        NEIGHBOR_MATCH_LENS_NUM);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());



}