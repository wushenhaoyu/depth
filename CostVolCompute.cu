#include "CostVolCompute.h"
#include "DataParameter.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>  // 提供CUDA图像处理函数（如cvtColor）
#include <opencv2/cudafilters.hpp>   // 提供CUDA滤波器（如Sobel）
#include <opencv2/cudaarithm.hpp>
#include <iostream>


using namespace cv;
using namespace cv::cuda;
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

// CUDA内核：计算成本体
__global__ void computeCostVolKernel(
    const float* d_inputImg, const float* d_gradImg, float* d_costVol,
    const int* d_pixelsMappingSet,
    const float2* d_lensCenterPoints,
    const MatchNeighborLens* d_matchNeighborLens,
    int width, int height, int channels,
    int xLensNum, int yLensNum,
    int xPixelBeginOffset, int yPixelBeginOffset,
    int disparityNum, float dispStep, float dispMin,
    int circleDiameter, int circleNarrow,
    int yCenterBeginOffset, int yCenterEndOffset,
    int xCenterBeginOffset, int xCenterEndOffset,
    int NEIGHBOR_MATCH_LENS_NUM_){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 当前像素点的索引
    int curCenterIndex = (y / circleDiameter) * xLensNum + (x / circleDiameter);
    int py = y - yPixelBeginOffset;
    int px = x - xPixelBeginOffset;

    // 检查是否属于当前微透镜的映射范围
    if (d_pixelsMappingSet[py * width + px] != curCenterIndex) return;

    const float2& centerPos = d_lensCenterPoints[curCenterIndex];
    const MatchNeighborLens* matchNeighborLens = &d_matchNeighborLens[curCenterIndex * NEIGHBOR_MATCH_LENS_NUM_];

    for (int d = 0; d < disparityNum; d++) {
        float realDisp = dispStep * d + dispMin;
        float tempSumCost = 0.0f;
        int tempCostNum = 0;

        for (int i = 0; i < NEIGHBOR_MATCH_LENS_NUM_; i++) {
            float matchCenterPos_y = matchNeighborLens[i].m_centerPosY;
            float matchCenterPos_x = matchNeighborLens[i].m_centerPosX;
            float centerDis = matchNeighborLens[i].m_centerDis;

            if (matchCenterPos_y < 0) break;

            // 计算匹配点的坐标
            float2 matchPoint;
            matchPoint.y = (centerDis + realDisp) * (matchCenterPos_y - centerPos.y) / centerDis + py;
            matchPoint.x = (centerDis + realDisp) * (matchCenterPos_x - centerPos.x) / centerDis + px;

            // 检查匹配点是否有效
            if (matchPoint.y < 0 || matchPoint.y >= height || matchPoint.x < 0 || matchPoint.x >= width) continue;

            int matchCenterIndex = matchNeighborLens[i].m_centerIndex;
            if (d_pixelsMappingSet[int(matchPoint.y) * width + int(matchPoint.x)] != matchCenterIndex) continue;

            // 计算当前点与匹配点的成本值
            float cost = 0.0f;
            if (int(matchPoint.y) == matchPoint.y && int(matchPoint.x) == matchPoint.x) {
                // 匹配点为整数像素位置
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
                cost = grdDiff; // 使用梯度差异作为成本值
            } else {
                // 匹配点为非整数像素位置，使用双线性插值
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
                cost = grdDiff; // 使用梯度差异作为成本值
            }

            tempSumCost += cost;
            tempCostNum++;
        }

        if (tempCostNum > 0) {
            tempSumCost /= tempCostNum;
        }

        // 存储成本值
        int costVolIndex = (py * width + px) * disparityNum + d;
        d_costVol[costVolIndex] = tempSumCost;
    }
}

void CostVolCompute::costVolDataCompute(const DataParameter& dataParameter, Mat* costVol) {
    // 初始化参数
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();

    int width = rawImageParameter.m_srcImgWidth;
    int height = rawImageParameter.m_srcImgHeight;
    int channels = 3; // 假设输入图像为RGB
    int disparityNum = disparityParameter.m_disNum;

    // 将输入图像转换为浮点格式
    Mat inputImgFloat;
    dataParameter.m_inputImg.convertTo(inputImgFloat, CV_32FC3, 1.0f / 255.0f);

    // 使用OpenCV CUDA计算灰度图和梯度图
    cuda::GpuMat d_inputImg, d_grayImg, d_gradX, d_gradY, d_gradImg;
    d_inputImg.upload(inputImgFloat);

    cuda::cvtColor(d_inputImg, d_grayImg, COLOR_BGR2GRAY);
    Ptr<cuda::Filter> sobelX = cuda::createSobelFilter(CV_32F, CV_32F, 1, 0, 3);
    Ptr<cuda::Filter> sobelY = cuda::createSobelFilter(CV_32F, CV_32F, 0, 1, 3);
    sobelX->apply(d_grayImg, d_gradX);
    sobelY->apply(d_grayImg, d_gradY);

    // 计算梯度幅度
    cv::cuda::addWeighted(d_gradX, 0.5, d_gradY, 0.5, 0.0, d_gradImg);

    // 分配GPU内存
    float* d_costVol;
    int* d_pixelsMappingSet;
    float2* d_lensCenterPoints;
    MatchNeighborLens* d_matchNeighborLens;

    cudaMalloc(&d_costVol, width * height * disparityNum * sizeof(float));
    cudaMalloc(&d_pixelsMappingSet, width * height * sizeof(int));
    cudaMalloc(&d_lensCenterPoints, rawImageParameter.m_xLensNum * rawImageParameter.m_yLensNum * sizeof(float2));
    cudaMalloc(&d_matchNeighborLens, rawImageParameter.m_xLensNum * rawImageParameter.m_yLensNum * NEIGHBOR_MATCH_LENS_NUM * sizeof(MatchNeighborLens));

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(d_pixelsMappingSet, microImageParameter.m_ppPixelsMappingSet[0], width * height * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lensCenterPoints, microImageParameter.m_ppLensCenterPoints[0], rawImageParameter.m_xLensNum * rawImageParameter.m_yLensNum * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matchNeighborLens, microImageParameter.m_ppMatchNeighborLens[0], rawImageParameter.m_xLensNum * rawImageParameter.m_yLensNum * NEIGHBOR_MATCH_LENS_NUM * sizeof(MatchNeighborLens), cudaMemcpyHostToDevice);

    // 配置线程块和网格大小
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 启动CUDA内核
    computeCostVolKernel<<<gridSize, blockSize>>>(
        (float*)d_inputImg.data, (float*)d_gradImg.data, d_costVol,
        d_pixelsMappingSet,
        d_lensCenterPoints,
        d_matchNeighborLens,
        width, height, channels,
        rawImageParameter.m_xLensNum, rawImageParameter.m_yLensNum,
        rawImageParameter.m_xPixelBeginOffset, rawImageParameter.m_yPixelBeginOffset,
        disparityParameter.m_disNum, disparityParameter.m_dispStep, disparityParameter.m_dispMin,
        microImageParameter.m_circleDiameter, microImageParameter.m_circleNarrow,
        rawImageParameter.m_yCenterBeginOffset, rawImageParameter.m_yCenterEndOffset,
        rawImageParameter.m_xCenterBeginOffset, rawImageParameter.m_xCenterEndOffset,
        NEIGHBOR_MATCH_LENS_NUM);

    // 将结果从设备内存复制回主机内存
    for (int d = 0; d < disparityNum; ++d) {
        size_t dataSize = width * height * sizeof(float);
        cudaMemcpy(costVol[d].data, d_costVol + d * width * height, dataSize, cudaMemcpyDeviceToHost);
    }

    // 释放设备内存
    cudaFree(d_costVol);
    cudaFree(d_pixelsMappingSet);
    cudaFree(d_lensCenterPoints);
    cudaFree(d_matchNeighborLens);
}