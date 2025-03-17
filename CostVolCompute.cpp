#include "CostVolCompute.h"
#include "DataParameter.h"

using namespace std;
using namespace cv;

CostVolCompute::CostVolCompute()
{
}

CostVolCompute::~CostVolCompute()
{
}

void CostVolCompute::costVolDataCompute(const DataParameter& dataParameter, Mat* costVol)
{
    // 初始化参数
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();

    // 将输入图像转换为浮点格式
    dataParameter.m_inputImg.convertTo(m_inputImg, CV_32FC3, 1.0f / 255.0f);

    // 计算灰度图和梯度图
    cv::Mat im_gray, tmp;
    cv::cvtColor(m_inputImg, tmp, COLOR_BGR2GRAY); // 转换为灰度图
    tmp.convertTo(im_gray, CV_32F, 1.0f / 255.0f); // 归一化到 [0, 1]
    cv::Sobel(im_gray, m_gradImg, CV_32F, 1, 0, 1); // 计算梯度
    m_gradImg += 0.5f; // 偏移梯度值以避免负值

    // 并行化计算成本体
    #pragma omp parallel for collapse(2)
    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
    {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
        {
            const Point2d& centerPos = microImageParameter.m_ppLensCenterPoints[y][x];
            const int curCenterIndex = y * rawImageParameter.m_xLensNum + x;

            // 遍历当前微透镜的像素范围
            for (int py = centerPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
                 py <= centerPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; py++)
            {
                for (int px = centerPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
                     px <= centerPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; px++)
                {
                    if (microImageParameter.m_ppPixelsMappingSet[py][px] == curCenterIndex)
                    {
                        // 遍历视差范围
                        for (int d = 0; d < disparityParameter.m_disNum; d++)
                        {
                            float* cost = (float*)costVol[d].ptr<float>(py - rawImageParameter.m_yPixelBeginOffset);

                            float tempSumCost = 0.0f;
                            int tempCostNum = 0;

                            // 计算当前视差对应的匹配点位置
                            float realDisp = disparityParameter.m_dispStep * d + disparityParameter.m_dispMin;
                            const MatchNeighborLens* matchNeighborLens = microImageParameter.m_ppMatchNeighborLens[y][x];

                            for (int i = 0; i < NEIGHBOR_MATCH_LENS_NUM; i++)
                            {
                                float matchCenterPos_y = matchNeighborLens[i].m_centerPosY;
                                float matchCenterPos_x = matchNeighborLens[i].m_centerPosX;
                                float centerDis = matchNeighborLens[i].m_centerDis;

                                if (matchCenterPos_y < 0)
                                    break;

                                // 计算匹配点的坐标
                                Point2d matchPoint;
                                matchPoint.y = (centerDis + realDisp) * (matchCenterPos_y - centerPos.y) / centerDis + py;
                                matchPoint.x = (centerDis + realDisp) * (matchCenterPos_x - centerPos.x) / centerDis + px;

                                int matchCenterIndex = matchNeighborLens[i].m_centerIndex;

                                // 检查匹配点是否有效
                                if (!isCurPointValid(matchPoint, matchCenterIndex, rawImageParameter, microImageParameter))
                                    continue;

                                // 计算当前点与匹配点的成本值
                                Point2d curPoint(px, py);
                                tempSumCost += bilinearInsertValue(curPoint, matchPoint);
                                tempCostNum++;
                            }

                            // 计算平均成本值
                            if (tempCostNum != 0)
                                tempSumCost /= tempCostNum;

                            // 存储成本值
                            cost[px - rawImageParameter.m_xPixelBeginOffset] = tempSumCost;
                        }
                    }
                }
            }
        }
    }
}

float CostVolCompute::bilinearInsertValue(const Point2d& curPoint, const Point2d& matchPoint)
{
    // 提前读取当前像素点的RGB和梯度值
    const float* curRGB = m_inputImg.ptr<float>(int(curPoint.y)) + 3 * int(curPoint.x);
    const float curGrad = m_gradImg.at<float>(int(curPoint.y), int(curPoint.x));

    if (int(matchPoint.y) == matchPoint.y && int(matchPoint.x) == matchPoint.x)
    {
        // 匹配点为整数像素位置
        const float* matchRGB = m_inputImg.ptr<float>(int(matchPoint.y)) + 3 * int(matchPoint.x);
        const float matchGrad = m_gradImg.at<float>(int(matchPoint.y), int(matchPoint.x));
        return myCostGrd(curRGB, matchRGB, &curGrad, &matchGrad);
    }
    else
    {
        // 匹配点为非整数像素位置，使用双线性插值
        int tempRx = int(matchPoint.x), tempRy = int(matchPoint.y);
        double alphaX = matchPoint.x - tempRx, alphaY = matchPoint.y - tempRy;

        float tempRc[3];
        const float* rgb_y1 = m_inputImg.ptr<float>(tempRy);
        const float* rgb_y2 = m_inputImg.ptr<float>(tempRy + 1);

        for (int i = 0; i < 3; i++)
        {
            tempRc[i] = (1 - alphaX) * (1 - alphaY) * rgb_y1[3 * tempRx + i] +
                        alphaX * (1 - alphaY) * rgb_y1[3 * (tempRx + 1) + i] +
                        (1 - alphaX) * alphaY * rgb_y2[3 * tempRx + i] +
                        alphaX * alphaY * rgb_y2[3 * (tempRx + 1) + i];
        }

        const float* grd_y1 = m_gradImg.ptr<float>(tempRy);
        const float* grd_y2 = m_gradImg.ptr<float>(tempRy + 1);
        float tempRg = (1 - alphaX) * (1 - alphaY) * grd_y1[tempRx] +
                       alphaX * (1 - alphaY) * grd_y1[tempRx + 1] +
                       (1 - alphaX) * alphaY * grd_y2[tempRx] +
                       alphaX * alphaY * grd_y2[tempRx + 1];

        return myCostGrd(curRGB, tempRc, &curGrad, &tempRg);
    }
}

bool CostVolCompute::isCurPointValid(const Point2d& matchPoint, int matchCenterIndex, 
                                     const RawImageParameter& rawImageParameter, 
                                     const MicroImageParameter& microImageParameter)
{
    // 检查是否在图像边界内
    if (matchPoint.y < 0 || matchPoint.y >= rawImageParameter.m_srcImgHeight ||
        matchPoint.x < 0 || matchPoint.x >= rawImageParameter.m_srcImgWidth)
        return false;

    // 检查是否属于当前微透镜的映射范围
    if (microImageParameter.m_ppPixelsMappingSet[int(matchPoint.y)][int(matchPoint.x)] != matchCenterIndex)
        return false;

    return true;
}

float CostVolCompute::myCostGrd(const float* lC, const float* rC, const float* lG, const float* rG)
{
    // 计算颜色差异
    float clrDiff = 0.0f;
    for (int c = 0; c < 3; c++)
    {
        clrDiff += fabs(lC[c] - rC[c]);
    }
    clrDiff *= 0.3333333333f; // 归一化

    // 计算梯度差异
    float grdDiff = fabs(lG[0] - rG[0]);

    // 返回梯度差异作为成本值
    return grdDiff;
}