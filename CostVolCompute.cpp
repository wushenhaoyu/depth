#include "CostVolCompute.h"
#include "DataParameter.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>

using namespace std;
using namespace cv;

CostVolCompute::CostVolCompute()
{
    // 初始化 CUDA 设备
    cuda::printCudaDeviceInfo(cuda::getDevice());
}

CostVolCompute::~CostVolCompute()
{
    // 释放资源
}

void CostVolCompute::costVolDataCompute(const DataParameter &dataParameter, Mat *costVol)
{
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();

    // 将输入图像转换为浮点类型
    Mat inputImg;
    dataParameter.m_inputImg.convertTo(inputImg, CV_32FC3, 1 / 255.0f);

    // 将图像上传到 GPU
    cuda::GpuMat d_inputImg(inputImg);
    cuda::GpuMat d_gradImg;

    // 计算灰度图像
    cuda::GpuMat d_grayImg;
    cuda::cvtColor(d_inputImg, d_grayImg, COLOR_RGB2GRAY);

    // 计算梯度图像
    Ptr<cuda::Filter> sobelFilter = cuda::createSobelFilter(CV_32F, CV_32F, 1, 0);
    sobelFilter->apply(d_grayImg, d_gradImg);

    // 将梯度图像下载到 CPU
    Mat gradImg;
    d_gradImg.download(gradImg);
    gradImg += 0.5;

    // 遍历每个像素并计算 cost
    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
    {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
        {
			std::cout << "cost --- y=" << y << "\tx=" << x << std::endl;
            costVolDataCompute(costVol, y, x, rawImageParameter, microImageParameter, disparityParameter, inputImg, gradImg);
        }
    }
}

void CostVolCompute::costVolDataCompute(cv::Mat *costVol, int y, int x, const RawImageParameter &rawImageParameter,
    const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter,
    const cv::Mat &inputImg, const cv::Mat &gradImg)
{
    Point2d &centerPos = microImageParameter.m_ppLensCenterPoints[y][x];
    int curCenterIndex = y * rawImageParameter.m_xLensNum + x;

    for (int py = centerPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
        py <= centerPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; py++)
    {
        for (int px = centerPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
            px <= centerPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; px++)
        {
            if (microImageParameter.m_ppPixelsMappingSet[py][px] == curCenterIndex) {
                for (int d = 0; d < disparityParameter.m_disNum; d++)
                {
                    float *cost = (float*)costVol[d].ptr<float>(py - rawImageParameter.m_yPixelBeginOffset);
                    cost[px - rawImageParameter.m_xPixelBeginOffset] = costVolDataCompute(y, x, py, px, d, rawImageParameter, microImageParameter, disparityParameter, inputImg, gradImg);
                }
            }
        }
    }
}

float CostVolCompute::costVolDataCompute(int y, int x, int py, int px, int d, const RawImageParameter &rawImageParameter,
    const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter,
    const cv::Mat &inputImg, const cv::Mat &gradImg)
{
    float tempSumCost = 0.0; int tempCostNum = 0;
    Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];
    Point2d matchPoint;
    float realDisp = disparityParameter.m_dispStep * d + disparityParameter.m_dispMin;
    MatchNeighborLens *matchNeighborLens = microImageParameter.m_ppMatchNeighborLens[y][x];

    for (int i = 0; i < NEIGHBOR_MATCH_LENS_NUM; i++)
    {
        float matchCenterPos_y = matchNeighborLens[i].m_centerPosY;
        float matchCenterPos_x = matchNeighborLens[i].m_centerPosX;
        float centerDis = matchNeighborLens[i].m_centerDis;
        if (matchCenterPos_y < 0)
            break;

        matchPoint.y = (centerDis + realDisp) * (matchCenterPos_y - curCenterPos.y) / centerDis + py;
        matchPoint.x = (centerDis + realDisp) * (matchCenterPos_x - curCenterPos.x) / centerDis + px;
        int matchCenterIndex = matchNeighborLens[i].m_centerIndex;

        if (!isCurPointValid(matchPoint, matchCenterIndex, rawImageParameter, microImageParameter))
            continue;
        Point2d curPoint(px, py);
        tempSumCost += bilinearInsertValue(curPoint, matchPoint, inputImg, gradImg);
        tempCostNum++;
    }

    if (tempCostNum != 0)
        tempSumCost /= tempCostNum;
    return tempSumCost;
}

float CostVolCompute::bilinearInsertValue(const Point2d &curPoint, const Point2d &matchPoint, const cv::Mat &inputImg, const cv::Mat &gradImg)
{
    float* lC = (float *)inputImg.ptr<float>(int(curPoint.y)) + 3 * int(curPoint.x);
    float* lG = (float *)gradImg.ptr<float>(int(curPoint.y)) + int(curPoint.x);

    if (int(matchPoint.y) == matchPoint.y && int(matchPoint.x) == matchPoint.x) {
        float* rC = (float *)inputImg.ptr<float>(int(matchPoint.y)) + 3 * int(matchPoint.x);
        float* rG = (float *)gradImg.ptr<float>(int(matchPoint.y)) + int(matchPoint.x);
        return myCostGrd(lC, rC, lG, rG);
    }
    else {
        int tempRx = int(matchPoint.x), tempRy = int(matchPoint.y);
        double alphaX = matchPoint.x - tempRx, alphaY = matchPoint.y - tempRy;

        float tempRc[3], tempRg;
        float *rgb_y1 = (float *)inputImg.ptr<float>(tempRy);
        float *rgb_y2 = (float *)inputImg.ptr<float>(tempRy + 1);
        for (int i = 0; i < 3; i++) {
            tempRc[i] = (1 - alphaX) * (1 - alphaY) * rgb_y1[3 * tempRx + i] + alphaX * (1 - alphaY) * rgb_y1[3 * (tempRx + 1) + i] +
                (1 - alphaX) * alphaY * rgb_y2[3 * tempRx + i] + alphaX * alphaY * rgb_y2[3 * (tempRx + 1) + i];
        }

        float *grd_y1 = (float *)gradImg.ptr<float>(tempRy);
        float *grd_y2 = (float *)gradImg.ptr<float>(tempRy + 1);
        tempRg = (1 - alphaX) * (1 - alphaY) * grd_y1[tempRx] + alphaX * (1 - alphaY) * grd_y1[tempRx + 1] +
            (1 - alphaX) * alphaY * grd_y2[tempRx] + alphaX * alphaY * grd_y2[tempRx + 1];

        float* rC = tempRc;
        float* rG = &tempRg;
        return myCostGrd(lC, rC, lG, rG);
    }
}

bool CostVolCompute::isCurPointValid(Point2d &matchPoint, int matchCenterIndex, const RawImageParameter &rawImageParameter,
    const MicroImageParameter &microImageParameter)
{
    float pm_y = matchPoint.y;
    float pm_x = matchPoint.x;

    if (pm_y < 0 || pm_y >= rawImageParameter.m_srcImgHeight || pm_x < 0 || pm_x >= rawImageParameter.m_srcImgWidth)
        return false;

    if (microImageParameter.m_ppPixelsMappingSet[int(pm_y)][int(pm_x)] != matchCenterIndex)
        return false;

    return true;
}

float CostVolCompute::myCostGrd(float* lC, float* rC, float* lG, float* rG)
{
    float clrDiff = 0;
    for (int c = 0; c < 3; c++) {
        float temp = fabs(lC[c] - rC[c]);
        clrDiff += temp;
    }
    clrDiff *= 0.3333333333;
    float grdDiff = fabs(lG[0] - rG[0]);
    return grdDiff;
}