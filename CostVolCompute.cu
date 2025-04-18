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


// CUDA 内核函数：将图像从 RGB 转换为灰度
__global__ void rgbToGrayKernel(float* d_inputImg,float* d_grayImg ,int width, int height) {
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
__global__ void sobelKernel( int width, int height,float* d_grayImg,float* d_gradImg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    if (x >= width || y >= height) return;

    // 获取当前像素及其邻居的索引
    int idx_north = (y - 1) * width + x;
    int idx_south = (y + 1) * width + x;
    int idx_west = y * width + (x - 1);
    int idx_east = y * width + (x + 1);
    int idx_north_west = (y - 1) * width + (x - 1);
    int idx_north_east = (y - 1) * width + (x + 1);
    int idx_south_west = (y + 1) * width + (x - 1);
    int idx_south_east = (y + 1) * width + (x + 1);

    // 计算水平梯度 (Gx)
    float Gx = -d_grayImg[idx_north_west] - 2 * d_grayImg[idx_west] - d_grayImg[idx_south_west]
               + d_grayImg[idx_north_east] + 2 * d_grayImg[idx_east] + d_grayImg[idx_south_east];

    // 存储水平梯度结果
    d_gradImg[idx] = Gx + 0.5;
}


__global__ void costVolDataComputeKernel(
    MicroImageParameterDevice *d_microImageParameter,
    float* d_inputImg,
    float* d_gradImg)
{
    // 当前处理的 patch 中心坐标
    int lensX = blockIdx.x + d_rawImageParameter.m_xCenterBeginOffset;
    int lensY = blockIdx.y + d_rawImageParameter.m_yCenterBeginOffset;

    if (lensX >= d_rawImageParameter.m_xLensNum - d_rawImageParameter.m_xCenterEndOffset ||
        lensY >= d_rawImageParameter.m_yLensNum - d_rawImageParameter.m_yCenterEndOffset)
        return;

    int lensIndex = lensY * d_rawImageParameter.m_xLensNum + lensX;
    CudaPoint2f centerPos = d_microImageParameter->m_ppLensCenterPoints[lensIndex];

    int patchX = threadIdx.x;
    int patchY = threadIdx.y;
    int dispId = threadIdx.z;

    int patchStrideX = blockDim.x;
    int patchStrideY = blockDim.y;
    int dispStrideZ = blockDim.z;

    int py_begin = int(centerPos.y - d_microImageParameter->m_circleDiameter / 2 + d_microImageParameter->m_circleNarrow);
    int py_end   = int(centerPos.y + d_microImageParameter->m_circleDiameter / 2 - d_microImageParameter->m_circleNarrow);
    int px_begin = int(centerPos.x - d_microImageParameter->m_circleDiameter / 2 + d_microImageParameter->m_circleNarrow);
    int px_end   = int(centerPos.x + d_microImageParameter->m_circleDiameter / 2 - d_microImageParameter->m_circleNarrow);

    for (int py = py_begin + patchY; py <= py_end; py += patchStrideY) {
        for (int px = px_begin + patchX; px <= px_end; px += patchStrideX) {
            int pixelIdx = py * d_rawImageParameter.m_srcImgWidth + px;
            if (d_microImageParameter->m_ppPixelsMappingSet[pixelIdx] != lensIndex)
                continue;

            for (int d = dispId; d < d_disparityParameter.m_disNum; d += dispStrideZ) {
                float tempSumCost = 0.0f;
                int tempCostNum = 0;

                CudaPoint2f curPoint(px, py);
                float realDisp = d_disparityParameter.m_dispStep * d + d_disparityParameter.m_dispMin;

                MatchNeighborLens* matchNeighborLens = &d_microImageParameter->m_ppMatchNeighborLens[lensIndex * NEIGHBOR_MATCH_LENS_NUM];

                for (int i = 0; i < NEIGHBOR_MATCH_LENS_NUM; i++) {
                    float mcy = matchNeighborLens[i].m_centerPosY;
                    float mcx = matchNeighborLens[i].m_centerPosX;
                    float cdis = matchNeighborLens[i].m_centerDis;
                    if (mcy < 0) break;

                    CudaPoint2f matchPoint;
                    matchPoint.y = (cdis + realDisp) * (mcy - centerPos.y) / cdis + py;
                    matchPoint.x = (cdis + realDisp) * (mcx - centerPos.x) / cdis + px;

                    int matchCenterIdx = matchNeighborLens[i].m_centerIndex;

                    if (matchPoint.y < 0 || matchPoint.y >= d_rawImageParameter.m_srcImgHeight ||
                        matchPoint.x < 0 || matchPoint.x >= d_rawImageParameter.m_srcImgWidth)
                        continue;

                    if (d_microImageParameter->m_ppPixelsMappingSet[int(matchPoint.y) * d_rawImageParameter.m_srcImgWidth + int(matchPoint.x)] != matchCenterIdx)
                        continue;

                    float lGrad = d_gradImg[py * d_rawImageParameter.m_srcImgWidth + px];

                    float rGrad;
                    int rx = int(matchPoint.x);
                    int ry = int(matchPoint.y);

                    if (matchPoint.x == rx && matchPoint.y == ry) {
                        rGrad = d_gradImg[ry * d_rawImageParameter.m_srcImgWidth + rx];
                    } else {
                        float alphaX = matchPoint.x - rx;
                        float alphaY = matchPoint.y - ry;
                        float* grd_y1 = d_gradImg + ry * d_rawImageParameter.m_srcImgWidth;
                        float* grd_y2 = d_gradImg + (ry + 1) * d_rawImageParameter.m_srcImgWidth;
                        rGrad = (1 - alphaX) * (1 - alphaY) * grd_y1[rx] +
                                alphaX * (1 - alphaY) * grd_y1[rx + 1] +
                                (1 - alphaX) * alphaY * grd_y2[rx] +
                                alphaX * alphaY * grd_y2[rx + 1];
                    }

                    tempSumCost += fabsf(lGrad - rGrad);
                    tempCostNum++;
                }

                if (tempCostNum > 0) {
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




// CUDA 加速版本的 costVolDataCompute 函数
void CostVolCompute::costVolDataCompute(const DataParameter &dataParameter, Mat *costVol)  {
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();



    // 计算CUDA网格和块的大小
    dim3 blockSize(32, 32);
    dim3 gridSize((rawImageParameter.m_srcImgWidth + blockSize.x - 1) / blockSize.x,
                  (rawImageParameter.m_srcImgHeight + blockSize.y - 1) / blockSize.y);


    // 调用CUDA内核函数：将图像从RGB转换为灰度
    /*rgbToGrayKernel<<<gridSize, blockSize>>>(d_inputImg,d_grayImg,rawImageParameter.m_srcImgWidth, rawImageParameter.m_srcImgHeight);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());



    // 调用CUDA内核函数：计算Sobel梯度
    sobelKernel<<<gridSize, blockSize>>>(rawImageParameter.m_srcImgWidth, rawImageParameter.m_srcImgHeight,d_grayImg,d_gradImg);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    saveSingleChannelGpuMemoryAsImage(d_gradImg, rawImageParameter.m_srcImgWidth , rawImageParameter.m_srcImgHeight, "./res/gpu_grad.png");

*/
    blockSize = dim3 (16, 16, 4);  // 每个线程块 256 threads，z 方向处理 disparity
    gridSize = dim3(
        rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterBeginOffset - rawImageParameter.m_xCenterEndOffset,
        rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterBeginOffset - rawImageParameter.m_yCenterEndOffset
    );

    // 调用 CUDA 内核函数
    costVolDataComputeKernel<<<gridSize, blockSize>>>(d_microImageParameter,d_inputImg,d_gradImg);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


}