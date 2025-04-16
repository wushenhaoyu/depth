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

__device__ float myCostGrd(const float* lC, const float* rC, const float* lG, const float* rG) {
    float clrDiff = 0.0f;
    /*for (int c = 0; c < 3; c++) {
        float temp = fabsf(lC[c] - rC[c]);
        clrDiff += temp;
    }
    clrDiff *= 0.3333333333f;*/

    float grdDiff = fabsf(lG[0] - rG[0]);
    //printf("clrDiff: %f, grdDiff: %f\n", clrDiff, grdDiff);

    return grdDiff;
}

__global__ void costVolDataComputeKernel(MicroImageParameterDevice *d_microImageParameter,float* d_inputImg,float* d_gradImg) {
    int y = blockIdx.y * blockDim.y + threadIdx.y + d_rawImageParameter.m_yCenterBeginOffset;
    int x = blockIdx.x * blockDim.x + threadIdx.x + d_rawImageParameter.m_xCenterBeginOffset;

    // 检查是否超出范围
    if (y >= d_rawImageParameter.m_yLensNum - d_rawImageParameter.m_yCenterEndOffset ||
        x >= d_rawImageParameter.m_xLensNum - d_rawImageParameter.m_xCenterEndOffset) {
        return;
    }

    // 计算一维索引
    int index = y * d_rawImageParameter.m_xLensNum + x;
    CudaPoint2f centerPos = d_microImageParameter->m_ppLensCenterPoints[index];
    //printf("x:%d y:%d centerPos.x:%d centerPos.y:%d\n", x, y, centerPos.x, centerPos.y);
    int curCenterIndex = index;
   
    int py_begin = int(centerPos.y - d_microImageParameter->m_circleDiameter / 2 + d_microImageParameter->m_circleNarrow);
    int py_end = int(centerPos.y + d_microImageParameter->m_circleDiameter / 2 - d_microImageParameter->m_circleNarrow);
    int px_begin = int(centerPos.x - d_microImageParameter->m_circleDiameter / 2 + d_microImageParameter->m_circleNarrow);
    int px_end = int(centerPos.x + d_microImageParameter->m_circleDiameter / 2 - d_microImageParameter->m_circleNarrow);
    

        for (int py = py_begin; 
         py <= py_end; py++) {
        for (int px = px_begin; 
             px <= px_end; px++) {
            int pyxIndex = py * d_rawImageParameter.m_srcImgWidth + px;
            if (d_microImageParameter->m_ppPixelsMappingSet[pyxIndex] == curCenterIndex) {
                for (int d = 0; d < d_disparityParameter.m_disNum; d++) {
                    float tempSumCost = 0.0f;
                    int tempCostNum = 0;
                

                    CudaPoint2f curPoint = CudaPoint2f(px, py);
                    CudaPoint2f matchPoint = CudaPoint2f(0.0f, 0.0f);
                    float realDisp = d_disparityParameter.m_dispStep * d + d_disparityParameter.m_dispMin;
                    MatchNeighborLens* matchNeighborLens = &d_microImageParameter->m_ppMatchNeighborLens[index * NEIGHBOR_MATCH_LENS_NUM];

                    for (int i = 0; i < NEIGHBOR_MATCH_LENS_NUM; i++) {
                        float matchCenterPos_y = matchNeighborLens[i].m_centerPosY;
                        float matchCenterPos_x = matchNeighborLens[i].m_centerPosX;
                        float centerDis = matchNeighborLens[i].m_centerDis;
                        /*if(d == 35 && py == 1839 && px == 797)
                            printf("i:%d,matchCenterPos_y:%f matchCenterPos_x:%f centerDis:%d\n", i,matchCenterPos_y, matchCenterPos_x, centerDis);
                        */
                        if (matchCenterPos_y < 0) break;

                        matchPoint.y = (centerDis + realDisp) * (matchCenterPos_y - centerPos.y) / centerDis + py;
                        matchPoint.x = (centerDis + realDisp) * (matchCenterPos_x - centerPos.x) / centerDis + px;
                        int matchCenterIndex = matchNeighborLens[i].m_centerIndex;

                        /*if(d == 35 && py == 1839 && px == 797)
                        printf("i:%d,matchPoint.y:%f,matchPoint.x:%f,matchCenterIndex:%d:\n",i,matchPoint.y,matchPoint.x,matchCenterIndex);
                        */

                        if (matchPoint.y < 0 || matchPoint.y >= d_rawImageParameter.m_srcImgHeight ||
                            matchPoint.x < 0 || matchPoint.x >= d_rawImageParameter.m_srcImgWidth ||
                            d_microImageParameter->m_ppPixelsMappingSet[int(matchPoint.y) * d_rawImageParameter.m_srcImgWidth + int(matchPoint.x)] != matchCenterIndex) continue;

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
                            tempRg = (1 - alphaX) * (1 - alphaY) * grd_y1[tempRx] +
                                     alphaX * (1 - alphaY) * grd_y1[tempRx + 1] +
                                     (1 - alphaX) * alphaY * grd_y2[tempRx] +
                                     alphaX * alphaY * grd_y2[tempRx + 1];

                            float* rC = tempRc;
                            float* rG = &tempRg;
                            if(d == 35 && py == 1839 && px == 797)
                            printf("i:%d,rC[0]:%f,rC[1]:%f,rC[2]:%f,rG:%f,lC[0]:%f,lC[1]:%f,lC[2]:%f,lG:%f,cache:%f\n",i,rC[0],rC[1],rC[2],*rG,lC[0],lC[1],lC[2],*lG,myCostGrd(lC, rC, lG, rG));
                            
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
                    if(d==35 && px == 144 &&py ==112)
                    {
                        printf("px:%d, py:%d, d:%d, tempSumCost: %f,res:%f\n", px, py, d, tempSumCost,d_costVol[35 * d_rawImageParameter.m_recImgHeight * d_rawImageParameter.m_recImgWidth + (112 - d_rawImageParameter.m_yPixelBeginOffset)* d_rawImageParameter.m_recImgWidth + 144 - d_rawImageParameter.m_xPixelBeginOffset]);
                    }

                    //if(d == 35 && py == 1839 )
                    //printf("px:%d, py:%d, d:%d, tempSumCost: %f\n", px, py, d, tempSumCost);
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
    blockSize = dim3 (32, 32);
    gridSize = dim3 (
        (rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterBeginOffset - rawImageParameter.m_xCenterEndOffset + blockSize.x - 1) / blockSize.x,
        (rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterBeginOffset - rawImageParameter.m_yCenterEndOffset + blockSize.y - 1) / blockSize.y
    );

    // 调用 CUDA 内核函数
    costVolDataComputeKernel<<<gridSize, blockSize>>>(d_microImageParameter,d_inputImg,d_gradImg);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


}