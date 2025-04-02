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

// CUDA 内核函数：将图像从 RGB 转换为灰度
__global__ void rgbToGrayKernel(int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width * 3 + x * 3;
    float r = d_inputImg[idx];
    float g = d_inputImg[idx + 1];
    float b = d_inputImg[idx + 2];
    //printf("%f %f %f\n", r, g, b);
    d_grayImg[y * width + x] = 0.299f * r + 0.587f * g + 0.114f * b;
    //printf("%f\n", d_grayImg[y * width + x]);
}

// CUDA 内核函数：计算 Sobel 梯度
__global__ void sobelKernel( int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // 获取相邻像素
    float gx = -d_grayImg[(y - 1) * width + (x - 1)] - 2 * d_grayImg[y * width + (x - 1)] - d_grayImg[(y + 1) * width + (x - 1)]
               + d_grayImg[(y - 1) * width + (x + 1)] + 2 * d_grayImg[y * width + (x + 1)] + d_grayImg[(y + 1) * width + (x + 1)];

    float gy = -d_grayImg[(y - 1) * width + (x - 1)] - d_grayImg[(y - 1) * width + x] - d_grayImg[(y - 1) * width + (x + 1)]
               + d_grayImg[(y + 1) * width + (x - 1)] + d_grayImg[(y + 1) * width + x] + d_grayImg[(y + 1) * width + (x + 1)];

    // 计算 Sobel 边缘响应
    d_gradImg[idx] = sqrtf(gx * gx + gy * gy) + 0.5;
}

__device__ float myCostGrd(const float* lC, const float* rC, const float* lG, const float* rG) {
    float clrDiff = 0.0f;
    for (int c = 0; c < 3; c++) {
        float temp = fabsf(lC[c] - rC[c]);
        clrDiff += temp;
    }
    clrDiff *= 0.3333333333f;

    float grdDiff = fabsf(lG[0] - rG[0]);

    return clrDiff + grdDiff;
}

__global__ void costVolDataComputeKernel() {
    int y = blockIdx.y * blockDim.y + threadIdx.y + d_rawImageParameter.m_yCenterBeginOffset;
    int x = blockIdx.x * blockDim.x + threadIdx.x + d_rawImageParameter.m_xCenterBeginOffset;

    // 检查是否超出范围
    if (y >= d_rawImageParameter.m_yLensNum - d_rawImageParameter.m_yCenterEndOffset ||
        x >= d_rawImageParameter.m_xLensNum - d_rawImageParameter.m_xCenterEndOffset) {
        return;
    }

    // 计算一维索引
    int index = y * d_rawImageParameter.m_xLensNum + x;

    CudaPoint2f centerPos = CudaPoint2f(d_microImageParameter.m_ppLensCenterPoints[index].x, d_microImageParameter.m_ppLensCenterPoints[index].y);
    int curCenterIndex = index;

    for (int py = centerPos.y - d_microImageParameter.m_circleDiameter / 2 + d_microImageParameter.m_circleNarrow; 
         py <= centerPos.y + d_microImageParameter.m_circleDiameter / 2 - d_microImageParameter.m_circleNarrow; py++) {
        for (int px = centerPos.x - d_microImageParameter.m_circleDiameter / 2 + d_microImageParameter.m_circleNarrow; 
             px <= centerPos.x + d_microImageParameter.m_circleDiameter / 2 - d_microImageParameter.m_circleNarrow; px++) {
            int pyxIndex = py * d_rawImageParameter.m_xLensNum + px;
            if (d_microImageParameter.m_ppPixelsMappingSet[pyxIndex] == curCenterIndex) {
                for (int d = 0; d < d_disparityParameter.m_disNum; d++) {
                    float tempSumCost = 0.0f;
                    int tempCostNum = 0;

                    CudaPoint2f curPoint = CudaPoint2f(px, py);
                    CudaPoint2f matchPoint = CudaPoint2f(0.0f, 0.0f);
                    float realDisp = d_disparityParameter.m_dispStep * d + d_disparityParameter.m_dispMin;
                    MatchNeighborLens* matchNeighborLens = &d_microImageParameter.m_ppMatchNeighborLens[index];

                    for (int i = 0; i < NEIGHBOR_MATCH_LENS_NUM; i++) {
                        float matchCenterPos_y = matchNeighborLens[i].m_centerPosY;
                        float matchCenterPos_x = matchNeighborLens[i].m_centerPosX;
                        float centerDis = matchNeighborLens[i].m_centerDis;

                        if (matchCenterPos_y < 0) break;

                        matchPoint.y = (centerDis + realDisp) * (matchCenterPos_y - centerPos.y) / centerDis + py;
                        matchPoint.x = (centerDis + realDisp) * (matchCenterPos_x - centerPos.x) / centerDis + px;
                        int matchCenterIndex = matchNeighborLens[i].m_centerIndex;

                        if (matchPoint.y < 0 || matchPoint.y >= d_rawImageParameter.m_srcImgHeight ||
                            matchPoint.x < 0 || matchPoint.x >= d_rawImageParameter.m_srcImgWidth ||
                            d_microImageParameter.m_ppPixelsMappingSet[int(matchPoint.y) * d_rawImageParameter.m_xLensNum + int(matchPoint.x)] != matchCenterIndex) continue;

                        float* lC = d_inputImg + py * d_rawImageParameter.m_srcImgWidth * 3 + px * 3;
                        float* lG = d_gradImg + py * d_rawImageParameter.m_srcImgWidth + px;

                        if (int(matchPoint.y) == matchPoint.y && int(matchPoint.x) == matchPoint.x) {
                            float* rC = d_inputImg + int(matchPoint.y) * d_rawImageParameter.m_srcImgWidth * 3 + int(matchPoint.x) * 3;
                            float* rG = d_gradImg + int(matchPoint.y) * d_rawImageParameter.m_srcImgWidth + int(matchPoint.x);
                            tempSumCost += myCostGrd(lC, rC, lG, rG);
                        } else {
                            int tempRx = int(matchPoint.x), tempRy = int(matchPoint.y);
                            double alphaX = matchPoint.x - tempRx, alphaY = matchPoint.y - tempRy;

                            float tempRc[3], tempRg;
                            float* rgb_y1 = d_inputImg + tempRy * d_rawImageParameter.m_srcImgWidth * 3;
                            float* rgb_y2 = d_inputImg + (tempRy + 1) * d_rawImageParameter.m_srcImgWidth * 3;
                            for (int i = 0; i < 3; i++) {
                                tempRc[i] = (1 - alphaX) * (1 - alphaY) * rgb_y1[tempRx * 3 + i] +
                                            alphaX * (1 - alphaY) * rgb_y1[(tempRx + 1) * 3 + i] +
                                            (1 - alphaX) * alphaY * rgb_y2[tempRx * 3 + i] +
                                            alphaX * alphaY * rgb_y2[(tempRx + 1) * 3 + i];
                            }

                            float* grd_y1 = d_gradImg + tempRy * d_rawImageParameter.m_srcImgWidth;
                            float* grd_y2 = d_gradImg + (tempRy + 1) * d_rawImageParameter.m_srcImgWidth;
                            tempRg = (1 - alphaX) * (1 - alphaY) * grd_y1[tempRx] +
                                     alphaX * (1 - alphaY) * grd_y1[tempRx + 1] +
                                     (1 - alphaX) * alphaY * grd_y2[tempRx] +
                                     alphaX * alphaY * grd_y2[tempRx + 1];

                            float* rC = tempRc;
                            float* rG = &tempRg;
                            tempSumCost += myCostGrd(lC, rC, lG, rG);
                        }
                        tempCostNum++;
                    }

                    if (tempCostNum != 0) {
                        tempSumCost /= tempCostNum;
                    }

                    int costVolIndex = d * d_rawImageParameter.m_srcImgHeight * d_rawImageParameter.m_srcImgWidth +
                                       (py - d_rawImageParameter.m_yPixelBeginOffset) * d_rawImageParameter.m_srcImgWidth +
                                       (px - d_rawImageParameter.m_xPixelBeginOffset);

                    printf("costVolIndex: %d, d: %d, py: %d, px: %d, value:%f \n", costVolIndex, d, py, px,tempSumCost);
                    d_costVol[costVolIndex] = tempSumCost;
                }
            }
        }
    }
}



// CUDA 加速版本的 costVolDataCompute 函数
void CostVolCompute::costVolDataCompute(const DataParameter &dataParameter, Mat *costVol)  {
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();



    // 计算CUDA网格和块的大小
    dim3 blockSize(32, 32);
    dim3 gridSize((rawImageParameter.m_srcImgWidth + blockSize.x - 1) / blockSize.x,
                  (rawImageParameter.m_srcImgHeight + blockSize.y - 1) / blockSize.y);

    /*float* d_data_ptr;
    CUDA_CHECK(cudaMemcpyFromSymbol(&d_data_ptr, d_inputImg, sizeof(float*)));
    float* h_data = new float[rawImageParameter.m_srcImgWidth * rawImageParameter.m_srcImgHeight * 3];
    CUDA_CHECK(cudaMemcpy(h_data, d_data_ptr, rawImageParameter.m_srcImgWidth * rawImageParameter.m_srcImgHeight * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    cv::Mat image(rawImageParameter.m_srcImgHeight, rawImageParameter.m_srcImgWidth, CV_32FC3, h_data);
    cv::Mat scaledImage;
    image.convertTo(scaledImage, CV_8UC3, 255.0);
    cv::imwrite("input.bmp", scaledImage);
    delete[] h_data;*/

    // 调用CUDA内核函数：将图像从RGB转换为灰度
    rgbToGrayKernel<<<gridSize, blockSize>>>(rawImageParameter.m_srcImgWidth, rawImageParameter.m_srcImgHeight);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    /*float* d_data_ptr;
    CUDA_CHECK(cudaMemcpyFromSymbol(&d_data_ptr, d_grayImg, sizeof(float*)));
    float* h_data = new float[rawImageParameter.m_srcImgWidth * rawImageParameter.m_srcImgHeight];
    CUDA_CHECK(cudaMemcpy(h_data, d_data_ptr, rawImageParameter.m_srcImgWidth * rawImageParameter.m_srcImgHeight * sizeof(float), cudaMemcpyDeviceToHost));
    cv::Mat image(rawImageParameter.m_srcImgHeight, rawImageParameter.m_srcImgWidth, CV_32FC1, h_data);
    cv::Mat scaledImage;
    cv::normalize(image, scaledImage, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("gray.png", scaledImage);
    delete[] h_data;*/


    // 调用CUDA内核函数：计算Sobel梯度
    sobelKernel<<<gridSize, blockSize>>>(rawImageParameter.m_srcImgWidth, rawImageParameter.m_srcImgHeight);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float* d_data_ptr;
    CUDA_CHECK(cudaMemcpyFromSymbol(&d_data_ptr, d_gradImg, sizeof(float*)));
    float* h_data = new float[rawImageParameter.m_srcImgWidth * rawImageParameter.m_srcImgHeight];
    CUDA_CHECK(cudaMemcpy(h_data, d_data_ptr, rawImageParameter.m_srcImgWidth * rawImageParameter.m_srcImgHeight * sizeof(float), cudaMemcpyDeviceToHost));
    cv::Mat image(rawImageParameter.m_srcImgHeight, rawImageParameter.m_srcImgWidth, CV_32FC1, h_data);
    cv::Mat scaledImage;
    cv::normalize(image, scaledImage, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("grad.png", scaledImage);
    delete[] h_data;

    blockSize = dim3 (32, 32);
    gridSize = dim3 (
        (rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterBeginOffset - rawImageParameter.m_xCenterEndOffset + blockSize.x - 1) / blockSize.x,
        (rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterBeginOffset - rawImageParameter.m_yCenterEndOffset + blockSize.y - 1) / blockSize.y
    );

    // 调用 CUDA 内核函数
    costVolDataComputeKernel<<<gridSize, blockSize>>>();

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


}