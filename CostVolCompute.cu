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
    float b = d_inputImg[idx];
    float g = d_inputImg[idx + 1];
    float r = d_inputImg[idx + 2];
    d_grayImg[y * width + x] = 0.299f * r + 0.587f * g + 0.114f * b; 
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
    //printf("clrDiff: %f, grdDiff: %f\n", clrDiff, grdDiff);

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
    //printf("index: %d,d_microImageParameter.m_ppPixelsMappingSet[pyxIndex]:%d\n", index, d_microImageParameter.m_ppPixelsMappingSet[index]);
    //printf("index: %d, centerPos: (%f, %f)\n",curCenterIndex, centerPos.x, centerPos.y);
    int py_begin = int(centerPos.y - d_microImageParameter.m_circleDiameter / 2 + d_microImageParameter.m_circleNarrow);
    int py_end = int(centerPos.y + d_microImageParameter.m_circleDiameter / 2 - d_microImageParameter.m_circleNarrow);
    int px_begin = int(centerPos.x - d_microImageParameter.m_circleDiameter / 2 + d_microImageParameter.m_circleNarrow);
    int px_end = int(centerPos.x + d_microImageParameter.m_circleDiameter / 2 - d_microImageParameter.m_circleNarrow);
    
   //printf("index: %d, centerPos: (%f, %f),d_microImageParameter.m_circleDiameter:%f,d_microImageParameter.m_circleNarrow:%f,res:%f\n",curCenterIndex, centerPos.x, centerPos.y,d_microImageParameter.m_circleDiameter,d_microImageParameter.m_circleNarrow,centerPos.y - d_microImageParameter.m_circleDiameter / 2 + d_microImageParameter.m_circleNarrow);
        //printf("d_microImageParameter.m_circleDiameter:%f,d_microImageParameter.m_circleNarrow:%f\n",d_microImageParameter.m_circleDiameter,d_microImageParameter.m_circleNarrow);
        for (int py = py_begin; 
         py <= py_end; py++) {
        for (int px = px_begin; 
             px <= px_end; px++) {
            int pyxIndex = py * d_rawImageParameter.m_srcImgWidth + px;
            if (d_microImageParameter.m_ppPixelsMappingSet[pyxIndex] == curCenterIndex) {
                //printf("index: %d,d_microImageParameter.m_ppPixelsMappingSet[pyxIndex]:%d\n", index, d_microImageParameter.m_ppPixelsMappingSet[pyxIndex]);
                for (int d = 0; d < d_disparityParameter.m_disNum; d++) {
                    float tempSumCost = 0.0f;
                    int tempCostNum = 0;
                

                    CudaPoint2f curPoint = CudaPoint2f(px, py);
                    CudaPoint2f matchPoint = CudaPoint2f(0.0f, 0.0f);
                    float realDisp = d_disparityParameter.m_dispStep * d + d_disparityParameter.m_dispMin;
                    MatchNeighborLens* matchNeighborLens = &d_microImageParameter.m_ppMatchNeighborLens[index * NEIGHBOR_MATCH_LENS_NUM];

                    for (int i = 0; i < NEIGHBOR_MATCH_LENS_NUM; i++) {
                        //if(y == 48 && x == 5 && d == 22 && px == 276)
                        //printf("i:%d, d:%d, py:%d, px:%d,y:%d,x:%d, matchNeighborLens[i].m_centerPosY:%f, matchNeighborLens[i].m_centerPosX:%f, matchNeighborLens[i].m_centerDis:%f\n", i, d, py, px,y,x,matchNeighborLens[i].m_centerPosY, matchNeighborLens[i].m_centerPosX, matchNeighborLens[i].m_centerDis);
                        float matchCenterPos_y = matchNeighborLens[i].m_centerPosY;
                        float matchCenterPos_x = matchNeighborLens[i].m_centerPosX;
                        float centerDis = matchNeighborLens[i].m_centerDis;

                        if (matchCenterPos_y < 0) break;

                        matchPoint.y = (centerDis + realDisp) * (matchCenterPos_y - centerPos.y) / centerDis + py;
                        matchPoint.x = (centerDis + realDisp) * (matchCenterPos_x - centerPos.x) / centerDis + px;
                        int matchCenterIndex = matchNeighborLens[i].m_centerIndex;

                        if (matchPoint.y < 0 || matchPoint.y >= d_rawImageParameter.m_srcImgHeight ||
                            matchPoint.x < 0 || matchPoint.x >= d_rawImageParameter.m_srcImgWidth ||
                            d_microImageParameter.m_ppPixelsMappingSet[int(matchPoint.y) * d_rawImageParameter.m_srcImgWidth + int(matchPoint.x)] != matchCenterIndex) continue;

                        float* lC = d_inputImg + py * d_rawImageParameter.m_srcImgWidth * 3 + px * 3;
                        float* lG = d_gradImg + py * d_rawImageParameter.m_srcImgWidth + px;

                        if (int(matchPoint.y) == matchPoint.y && int(matchPoint.x) == matchPoint.x) {
                            float* rC = d_inputImg + int(matchPoint.y) * d_rawImageParameter.m_srcImgWidth * 3 + int(matchPoint.x) * 3;
                            float* rG = d_gradImg + int(matchPoint.y) * d_rawImageParameter.m_srcImgWidth + int(matchPoint.x);
                            tempSumCost += myCostGrd(lC, rC, lG, rG);
                        } else {
                            int tempRx = int(matchPoint.x), tempRy = int(matchPoint.y);
                            float alphaX = matchPoint.x - tempRx, alphaY = matchPoint.y - tempRy;

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
                            if(px == 970 && py == 1839 && x == 21 && y == 48 && d == 35 && i == 0 )
                               {
                                    printf("%f , %f \n", rgb_y1[0], rgb_y2[0]);
                               }
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

                    int costVolIndex = d * d_rawImageParameter.m_recImgHeight * d_rawImageParameter.m_recImgWidth +
                    (py - d_rawImageParameter.m_yPixelBeginOffset) * d_rawImageParameter.m_recImgWidth +
                    (px - d_rawImageParameter.m_xPixelBeginOffset);
                    d_costVol[costVolIndex] = tempSumCost;
                }
            }
        }
    }
}


__global__ void initializeCostVolume(int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算全局索引
    if (idx < totalElements) {
        d_costVol[idx] = 0.0f; // 将每个元素初始化为0
        float temp = d_costVol[idx];
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
    
    dataParameter.m_inputImg.convertTo(m_inputImg, CV_32FC3, 1 / 255.0f);
    cv::Mat im_gray, tmp;
    //m_inputImg.convertTo(tmp, CV_32F);
    //cv::cvtColor(tmp, im_gray, COLOR_RGB2GRAY);

    // Compare GPU and CPU grayscale images
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

    /*cv::Sobel(im_gray, m_gradImg, CV_32F, 1, 0, 1);
    m_gradImg += 0.5;

    float* d_data_ptr;
    CUDA_CHECK(cudaMemcpyFromSymbol(&d_data_ptr, d_gradImg, sizeof(float*)));
    float* h_data = new float[rawImageParameter.m_srcImgWidth * rawImageParameter.m_srcImgHeight];
    CUDA_CHECK(cudaMemcpy(h_data, d_data_ptr, rawImageParameter.m_srcImgWidth * rawImageParameter.m_srcImgHeight * sizeof(float), cudaMemcpyDeviceToHost));
    cv::Mat gpu_grad(rawImageParameter.m_srcImgHeight, rawImageParameter.m_srcImgWidth, CV_32FC1, h_data);

    cv::Mat cpu_grad;
    m_gradImg.convertTo(cpu_grad, CV_32F);

    cv::Mat diff;
    cv::absdiff(cpu_grad, gpu_grad, diff);
    cv::normalize(diff, diff, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::imwrite("gpu_grad.png", gpu_grad * 255.0);
    cv::imwrite("cpu_grad.png", cpu_grad * 255.0);
    cv::imwrite("diff_grad.png", diff);

    delete[] h_data;*/



    /*float* d_data_ptr;
    CUDA_CHECK(cudaMemcpyFromSymbol(&d_data_ptr, d_gradImg, sizeof(float*)));
    float* h_data = new float[rawImageParameter.m_srcImgWidth * rawImageParameter.m_srcImgHeight];
    CUDA_CHECK(cudaMemcpy(h_data, d_data_ptr, rawImageParameter.m_srcImgWidth * rawImageParameter.m_srcImgHeight * sizeof(float), cudaMemcpyDeviceToHost));
    cv::Mat image(rawImageParameter.m_srcImgHeight, rawImageParameter.m_srcImgWidth, CV_32FC1, h_data);
    cv::Mat scaledImage;
    cv::normalize(image, scaledImage, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("grad.png", scaledImage);
    delete[] h_data;*/

    size_t totalElements = disparityParameter.m_disNum * rawImageParameter.m_recImgHeight * rawImageParameter.m_recImgWidth;
	int totalElements_ = disparityParameter.m_disNum * rawImageParameter.m_recImgHeight * rawImageParameter.m_recImgWidth;
	int blockSize_ = 256; // 每个线程块的线程数
    int numBlocks_ = (totalElements + blockSize_ - 1) / blockSize_; // 计算所需的线程块数量
    initializeCostVolume<<<numBlocks_, blockSize>>>(totalElements_);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

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