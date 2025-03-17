#include "ConfidenceCompute.h"
#include "DataParameter.h"
#include <opencv2/cudaimgproc.hpp>  // 提供CUDA图像处理函数（如cvtColor）
#include <opencv2/cudafilters.hpp>   // 提供CUDA滤波器（如Sobel）
#include <opencv2/cudaarithm.hpp>
using namespace std;
using namespace cv;

#define GRADIDENT_THRESHOLD  70//梯度区域阈值  //45~70:70
#define CIRCLE_GRAD_POINT_NUM 0 //小圆内有效梯度点的数目
#define MNN_CONFIDENT_MEASURE_THRES 10.0 //MNN指标置信度阈值   10
#define FINAL_CONFIDENT_MEASURE_THRES 35.0 //置信度二值化分割时的阈值 35
#define PI 3.1415926535898f

//#define RADIUS 15.5
#define RADIUS 8

ConfidenceCompute::ConfidenceCompute()
	:m_pGradientCircleMask(nullptr), m_pConfidentMask(nullptr)
{

}

ConfidenceCompute::~ConfidenceCompute()
{
	if (m_pGradientCircleMask != nullptr)
		delete m_pGradientCircleMask;
	if (m_pConfidentMask != nullptr)
		delete m_pConfidentMask;
}



// CUDA内核：梯度滤波
__global__ void gradientFilterKernel(
    const float* src_grad, float* dst_grad, const float* divideMask,
    int width, int height, int pitch,
    int microImageWidth, int microImageHeight,
    int xPixelBeginOffset, int yPixelBeginOffset,
    int circleDiameter, int circleNarrow,
    const Point2d* lensCenterPoints,
    int xCenterBeginOffset, int yCenterBeginOffset,
    int xCenterEndOffset, int yCenterEndOffset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 遍历微透镜中心
    for (int ly = yCenterBeginOffset; ly < microImageHeight - yCenterEndOffset; ly++)
    {
        for (int lx = xCenterBeginOffset; lx < microImageWidth - xCenterEndOffset; lx++)
        {
            Point2d centerPos = lensCenterPoints[ly * microImageWidth + lx];
            int x_begin = centerPos.x - xPixelBeginOffset - circleDiameter / 2 + circleNarrow;
            int y_begin = centerPos.y - yPixelBeginOffset - circleDiameter / 2 + circleNarrow;
            int x_end = centerPos.x - xPixelBeginOffset + circleDiameter / 2 - circleNarrow;
            int y_end = centerPos.y - yPixelBeginOffset + circleDiameter / 2 - circleNarrow;

            if (x >= x_begin && x <= x_end && y >= y_begin && y <= y_end)
            {
                // 计算梯度滤波
                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ky++)
                {
                    for (int kx = -1; kx <= 1; kx++)
                    {
                        int nx = x + kx;
                        int ny = y + ky;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                        {
                            sum += src_grad[ny * pitch + nx];
                        }
                    }
                }
                dst_grad[y * pitch + x] = sum / divideMask[(y - y_begin) * (x_end - x_begin + 1) + (x - x_begin)];
            }
        }
    }
}

// CUDA内核：MMN置信度计算
__global__ void computeMMNConfidence(
    const float* costVol, float* confidentMat, float* confidentMat2,
    int width, int height, int disparityNum,
    int microImageWidth, int microImageHeight,
    int xPixelBeginOffset, int yPixelBeginOffset,
    int circleDiameter, int circleNarrow,
    const Point2d* lensCenterPoints,
    int xCenterBeginOffset, int yCenterBeginOffset,
    int xCenterEndOffset, int yCenterEndOffset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 遍历微透镜中心
    for (int ly = yCenterBeginOffset; ly < microImageHeight - yCenterEndOffset; ly++)
    {
        for (int lx = xCenterBeginOffset; lx < microImageWidth - xCenterEndOffset; lx++)
        {
            Point2d centerPos = lensCenterPoints[ly * microImageWidth + lx];
            int x_begin = centerPos.x - xPixelBeginOffset - circleDiameter / 2 + circleNarrow;
            int y_begin = centerPos.y - yPixelBeginOffset - circleDiameter / 2 + circleNarrow;
            int x_end = centerPos.x - xPixelBeginOffset + circleDiameter / 2 - circleNarrow;
            int y_end = centerPos.y - yPixelBeginOffset + circleDiameter / 2 - circleNarrow;

            if (x >= x_begin && x <= x_end && y >= y_begin && y <= y_end)
            {
                float dCostMin = FLT_MAX;
                float dCostSec = FLT_MAX;
                float sumCost = 0.0f;

                for (int d = 0; d < disparityNum; d++)
                {
                    float cost = costVol[d * width * height + y * width + x];
                    if (cost < dCostMin)
                    {
                        dCostSec = dCostMin;
                        dCostMin = cost;
                    }
                    else if (cost > dCostMin && cost < dCostSec)
                    {
                        dCostSec = cost;
                    }
                    sumCost += cost;
                }

                float confMMN = (dCostSec - dCostMin) / sumCost;
                confidentMat[y * width + x] = confMMN;
                confidentMat2[y * width + x] = confMMN;
            }
        }
    }
}

void ConfidenceCompute::confidenceMeasureCompute(
    const DataParameter& dataParameter, cv::Mat*& costVol)
{
    // 提取参数
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();
    FilterPatameter filterPatameter = dataParameter.getFilterPatameter();
    m_folderPath = dataParameter.m_folderPath;

    // 将输入图像转换为浮点类型
    cv::Mat inputRecImg;
    dataParameter.m_inputImgRec.convertTo(inputRecImg, CV_32FC3);

    // 梯度指标计算（使用CUDA）
    cv::cuda::GpuMat gpu_inputRecImg, gpu_im_gray, gpu_dst_x, gpu_dst_y, gpu_dst;
    gpu_inputRecImg.upload(inputRecImg);
    cv::cuda::cvtColor(gpu_inputRecImg, gpu_im_gray, COLOR_RGB2GRAY);
    Ptr<cuda::Filter> sobelX = cuda::createSobelFilter(CV_32F, CV_32F, 1, 0, 3);
    Ptr<cuda::Filter> sobelY = cuda::createSobelFilter(CV_32F, CV_32F, 0, 1, 3);

    // 应用Sobel滤波器
    sobelX->apply(gpu_im_gray, gpu_dst_x);
    sobelY->apply(gpu_im_gray, gpu_dst_y);
    cv::cuda::addWeighted(gpu_dst_x, 0.5, gpu_dst_y, 0.5, 0, gpu_dst);

    cv::cuda::GpuMat gpu_dst_abs;
    cv::cuda::abs(gpu_dst, gpu_dst_abs);

    // 将结果转换为8位图像
    cv::cuda::GpuMat gpu_dst_8u;
    gpu_dst_abs.convertTo(gpu_dst_8u, CV_8UC1);
    // 将梯度图传回CPU
    cv::Mat src_grad;
    gpu_dst.download(src_grad);

    //std::cout<<"梯度图计算完成"<<std::endl;

    // CUDA内存分配
    float* d_src_grad;
    float* d_dst_grad;
    float* d_divideMask;
    cudaMalloc(&d_src_grad, src_grad.total() * sizeof(float));
    cudaMalloc(&d_dst_grad, src_grad.total() * sizeof(float));
    cudaMalloc(&d_divideMask, src_grad.total() * sizeof(float));

    cudaMemcpy(d_src_grad, src_grad.ptr<float>(), src_grad.total() * sizeof(float), cudaMemcpyHostToDevice);

    // 遍历微透镜中心进行梯度滤波（CUDA内核）
    dim3 blockSize(16, 16);
    dim3 gridSize((src_grad.cols + blockSize.x - 1) / blockSize.x, (src_grad.rows + blockSize.y - 1) / blockSize.y);

    gradientFilterKernel<<<gridSize, blockSize>>>(
        d_src_grad, d_dst_grad, d_divideMask,
        src_grad.cols, src_grad.rows, src_grad.step,
        rawImageParameter.m_xLensNum, rawImageParameter.m_yLensNum,
        rawImageParameter.m_xPixelBeginOffset, rawImageParameter.m_yPixelBeginOffset,
        microImageParameter.m_circleDiameter, microImageParameter.m_circleNarrow,
        *microImageParameter.m_ppLensCenterPoints,
        rawImageParameter.m_xCenterBeginOffset, rawImageParameter.m_yCenterBeginOffset,
        rawImageParameter.m_xCenterEndOffset, rawImageParameter.m_yCenterEndOffset);

    cudaDeviceSynchronize();



    // 将结果传回CPU
    cv::Mat dst_grad(src_grad.size(), CV_32FC1);
    cudaMemcpy(dst_grad.ptr<float>(), d_dst_grad, dst_grad.total() * sizeof(float), cudaMemcpyDeviceToHost);

    cv::Mat outPutImg;
        // 第一次使用 minVal 和 maxVal
    double minVal, maxVal;
    minMaxLoc(dst_grad, &minVal, &maxVal);
    Mat dst_grad2;
    dst_grad.convertTo(dst_grad2, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    cv::threshold(dst_grad2, outPutImg, GRADIDENT_THRESHOLD, 255, THRESH_BINARY);

    // 梯度圆 mask 标记
    m_pGradientCircleMask = new cv::Mat;
    *m_pGradientCircleMask = cv::Mat::zeros(rawImageParameter.m_yLensNum, rawImageParameter.m_xLensNum, CV_8UC1);

    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
    {
        uchar *yCircleMask = (uchar*)(*m_pGradientCircleMask).ptr<uchar>(y);
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
        {
            Point2d &centerPos = microImageParameter.m_ppLensCenterPoints[y][x];
            int curCenterIndex = y * rawImageParameter.m_xLensNum + x, sumCount = 0;

            for (int py = centerPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
                py <= centerPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; py++)
            {
                uchar *pYMask = (uchar *)outPutImg.ptr<uchar>(py - rawImageParameter.m_yPixelBeginOffset);
                for (int px = centerPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
                    px <= centerPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; px++)
                {
                    if (microImageParameter.m_ppPixelsMappingSet[py][px] == curCenterIndex)
                    {
                        if (pYMask[px - rawImageParameter.m_xPixelBeginOffset] > 20)
                            ++sumCount;
                    }
                }
            }

            if (sumCount > CIRCLE_GRAD_POINT_NUM)
                yCircleMask[x] = 255;
        }
    }

    // 保存梯度结果
    cv::imwrite(m_folderPath + "/gradientMeasureMask.png", outPutImg);


    // MMN置信度计算（CUDA内核）
    float* d_costVol;
    float* d_confidentMat;
    float* d_confidentMat2;
    cudaMalloc(&d_costVol, costVol->total() * sizeof(float));
    cudaMalloc(&d_confidentMat, src_grad.total() * sizeof(float));
    cudaMalloc(&d_confidentMat2, src_grad.total() * sizeof(float));

    cudaMemcpy(d_costVol, costVol[0].ptr<float>(), costVol->total() * sizeof(float), cudaMemcpyHostToDevice);

    computeMMNConfidence<<<gridSize, blockSize>>>(
        d_costVol, d_confidentMat, d_confidentMat2,
        src_grad.cols, src_grad.rows, disparityParameter.m_disNum,
        rawImageParameter.m_xLensNum, rawImageParameter.m_yLensNum,
        rawImageParameter.m_xPixelBeginOffset, rawImageParameter.m_yPixelBeginOffset,
        microImageParameter.m_circleDiameter, microImageParameter.m_circleNarrow,
        *microImageParameter.m_ppLensCenterPoints,
        rawImageParameter.m_xCenterBeginOffset, rawImageParameter.m_yCenterBeginOffset,
        rawImageParameter.m_xCenterEndOffset, rawImageParameter.m_yCenterEndOffset);

    cudaDeviceSynchronize();

    // 将置信度结果传回CPU
    cv::Mat confidentMat(src_grad.size(), CV_32FC1);
    cv::Mat confidentMat2(src_grad.size(), CV_32FC1);
    cudaMemcpy(confidentMat.ptr<float>(), d_confidentMat, confidentMat.total() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(confidentMat2.ptr<float>(), d_confidentMat2, confidentMat2.total() * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放CUDA内存
    cudaFree(d_src_grad);
    cudaFree(d_dst_grad);
    cudaFree(d_divideMask);
    cudaFree(d_costVol);
    cudaFree(d_confidentMat);
    cudaFree(d_confidentMat2);

    // 保存结果
    cv::Mat conftmp, conftmp2;
    confidentMat.convertTo(conftmp, CV_8UC1);
    string storeName = m_folderPath + "/conf_measure.png";
    cv::imwrite(storeName, conftmp);

    minMaxLoc(confidentMat2, &minVal, &maxVal);
    confidentMat2.convertTo(conftmp2, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    storeName = m_folderPath + "/final_conf_measure.png";
    cv::imwrite(storeName, conftmp2);

    storeName = m_folderPath + "/ConfidentMask.png";
    setConfidentMask(conftmp2, storeName, rawImageParameter, microImageParameter);

    cv::Mat mask = cv::Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_8UC1);
    confidentCircleJudge(conftmp2, mask, rawImageParameter, microImageParameter);
    drawConfidentCircle(costVol, mask, rawImageParameter, microImageParameter, disparityParameter, inputRecImg);

    std::string confidentImgStoreName = m_folderPath + "/confident_circle.png";
    lowTextureAreaPlot(rawImageParameter, microImageParameter, conftmp2, mask, confidentImgStoreName, CircleDrawMode::e_gray, true);
}


/*void ConfidenceCompute::confidenceMeasureCompute(
    const DataParameter &dataParameter, 
    cv::Mat *&costVol)
{
    // 提取参数
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();
    FilterPatameter filterPatameter = dataParameter.getFilterPatameter();
    m_folderPath = dataParameter.m_folderPath;

    // 将输入图像转换为浮点类型
    cv::Mat inputRecImg;
    dataParameter.m_inputImgRec.convertTo(inputRecImg, CV_32FC3);

    // 梯度指标计算
    cv::Mat im_gray, dst_x, dst_y, dst, outPutImg;
    cvtColor(inputRecImg, im_gray, COLOR_RGB2GRAY);
    Sobel(im_gray, dst_x, CV_32F, 1, 0);
    Sobel(im_gray, dst_y, CV_32F, 0, 1);
    addWeighted(dst_x, 0.5, dst_y, 0.5, 0, dst);
    convertScaleAbs(dst, dst);
    cv::Mat src_grad = dst.clone();
    cv::Mat dst_grad = cv::Mat::zeros(src_grad.rows, src_grad.cols, CV_32FC1);
    src_grad.convertTo(src_grad, CV_32FC1);
    src_grad = src_grad.mul((*filterPatameter.m_pValidPixelsMask)(cv::Rect(
        rawImageParameter.m_xPixelBeginOffset, rawImageParameter.m_yPixelBeginOffset,
        rawImageParameter.m_recImgWidth, rawImageParameter.m_recImgHeight)));

    // 遍历每个微透镜中心进行梯度滤波
    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
    {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
        {
            Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];
            int x_begin = curCenterPos.x - rawImageParameter.m_xPixelBeginOffset - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
            int y_begin = curCenterPos.y - rawImageParameter.m_yPixelBeginOffset - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
            int x_end = curCenterPos.x - rawImageParameter.m_xPixelBeginOffset + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow;
            int y_end = curCenterPos.y - rawImageParameter.m_yPixelBeginOffset + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow;

            cv::Mat divideMask = (*filterPatameter.m_pValidNeighborPixelsNum)(cv::Rect(
                x_begin + rawImageParameter.m_xPixelBeginOffset,
                y_begin + rawImageParameter.m_yPixelBeginOffset,
                x_end - x_begin + 1, y_end - y_begin + 1));
            cv::Mat srcCost = src_grad(cv::Rect(x_begin, y_begin, x_end - x_begin + 1, y_end - y_begin + 1));
            cv::Mat destCost = dst_grad(cv::Rect(x_begin, y_begin, x_end - x_begin + 1, y_end - y_begin + 1));

            cv::filter2D(srcCost, destCost, -1, filterPatameter.m_filterKnernel, cv::Point(2, 2), 0, BORDER_CONSTANT);
            cv::divide(destCost, divideMask, destCost);
        }
    }

    dst_grad = dst_grad.mul((*filterPatameter.m_pValidPixelsMask)(cv::Rect(
        rawImageParameter.m_xPixelBeginOffset, rawImageParameter.m_yPixelBeginOffset,
        rawImageParameter.m_recImgWidth, rawImageParameter.m_recImgHeight)));

    // 第一次使用 minVal 和 maxVal
    cv::Mat outPutImg;
    double minVal, maxVal;
    minMaxLoc(dst_grad, &minVal, &maxVal);
    Mat dst_grad2;
    dst_grad.convertTo(dst_grad2, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    cv::threshold(dst_grad2, outPutImg, GRADIDENT_THRESHOLD, 255, THRESH_BINARY);

    // 梯度圆 mask 标记
    m_pGradientCircleMask = new cv::Mat;
    *m_pGradientCircleMask = cv::Mat::zeros(rawImageParameter.m_yLensNum, rawImageParameter.m_xLensNum, CV_8UC1);

    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
    {
        uchar *yCircleMask = (uchar*)(*m_pGradientCircleMask).ptr<uchar>(y);
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
        {
            Point2d &centerPos = microImageParameter.m_ppLensCenterPoints[y][x];
            int curCenterIndex = y * rawImageParameter.m_xLensNum + x, sumCount = 0;

            for (int py = centerPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
                py <= centerPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; py++)
            {
                uchar *pYMask = (uchar *)outPutImg.ptr<uchar>(py - rawImageParameter.m_yPixelBeginOffset);
                for (int px = centerPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
                    px <= centerPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; px++)
                {
                    if (microImageParameter.m_ppPixelsMappingSet[py][px] == curCenterIndex)
                    {
                        if (pYMask[px - rawImageParameter.m_xPixelBeginOffset] > 20)
                            ++sumCount;
                    }
                }
            }

            if (sumCount > CIRCLE_GRAD_POINT_NUM)
                yCircleMask[x] = 255;
        }
    }

    // 保存梯度结果
    cv::imwrite(m_folderPath + "/gradientMeasureMask.png", outPutImg);

    // MMN 置信度计算
    cv::Mat confidentMat = cv::Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_32FC1);
    cv::Mat confidentMat2 = cv::Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_32FC1);

    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
    {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
        {
            Point2d &centerPos = microImageParameter.m_ppLensCenterPoints[y][x];
            int curCenterIndex = y * rawImageParameter.m_xLensNum + x;

            float picConfMin = FLT_MAX, picConfMax = FLT_MIN;
            for (int py = centerPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
                py <= centerPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; py++)
            {
                for (int px = centerPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
                    px <= centerPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; px++)
                {
                    float *yConf = confidentMat.ptr<float>(py - rawImageParameter.m_yPixelBeginOffset);
                    float *yConf2 = confidentMat2.ptr<float>(py - rawImageParameter.m_yPixelBeginOffset);
                    float dCostMin = FLT_MAX;
                    float dCostSec = FLT_MAX;
                    float sumCost = 0.0;
                    if (microImageParameter.m_ppPixelsMappingSet[py][px] == curCenterIndex)
                    {
                        for (int d = 0; d < disparityParameter.m_disNum; d++)
                        {
                            float *cost = (float*)costVol[d].ptr<float>(py - rawImageParameter.m_yPixelBeginOffset);
                            if (cost[px - rawImageParameter.m_xPixelBeginOffset] < dCostMin)
                            {
                                dCostSec = dCostMin;
                                dCostMin = cost[px - rawImageParameter.m_xPixelBeginOffset];
                            }
                            else if (cost[px - rawImageParameter.m_xPixelBeginOffset] > dCostMin && cost[px - rawImageParameter.m_xPixelBeginOffset] < dCostSec)
                                dCostSec = cost[px - rawImageParameter.m_xPixelBeginOffset];

                            sumCost += cost[px - rawImageParameter.m_xPixelBeginOffset];
                        }
                    }
                    float confMMN = (dCostSec - dCostMin) / sumCost; // MMN置信度
                    if (confMMN > picConfMax) picConfMax = confMMN;
                    if (confMMN < picConfMin) picConfMin = confMMN;

                    yConf[px - rawImageParameter.m_xPixelBeginOffset] = confMMN;
                    yConf2[px - rawImageParameter.m_xPixelBeginOffset] = confMMN;
                }
            }

            for (int py = centerPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
                py <= centerPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; py++)
            {
                for (int px = centerPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
                    px <= centerPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; px++)
                {
                    float *yConf = confidentMat.ptr<float>(py - rawImageParameter.m_yPixelBeginOffset);
                    if (microImageParameter.m_ppPixelsMappingSet[py][px] == curCenterIndex)
                    {
                        yConf[px - rawImageParameter.m_xPixelBeginOffset] = (yConf[px - rawImageParameter.m_xPixelBeginOffset] - picConfMin) / (picConfMax - picConfMin) * 255.0;
                    }
                }
            }
        }
    }

    // 保存置信度结果
    cv::Mat conftmp, conftmp2;
    confidentMat.convertTo(conftmp, CV_8UC1);
    string storeName = m_folderPath + "/conf_measure.png";
    cv::imwrite(storeName, conftmp);

    // 第二次使用 minVal 和 maxVal（复用变量）
    minMaxLoc(confidentMat2, &minVal, &maxVal);
    confidentMat2.convertTo(conftmp2, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    storeName = m_folderPath + "/final_conf_measure.png";
    cv::imwrite(storeName, conftmp2);

    storeName = m_folderPath + "/ConfidentMask.png";
    setConfidentMask(conftmp2, storeName, rawImageParameter, microImageParameter);

    Mat mask = Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_8UC1);
    confidentCircleJudge(conftmp2, mask, rawImageParameter, microImageParameter);
    drawConfidentCircle(costVol, mask, rawImageParameter, microImageParameter, disparityParameter, inputRecImg);

    std::string confidentImgStoreName = m_folderPath + "/confident_circle.png";
    lowTextureAreaPlot(rawImageParameter, microImageParameter, conftmp2, mask, confidentImgStoreName, CircleDrawMode::e_gray, true);
}*/

void ConfidenceCompute::setConfidentMask(cv::Mat &confidentMat, std::string confidentMaskName, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter)
{//设置置信度的mask
	m_pConfidentMask = new cv::Mat(confidentMat.rows, confidentMat.cols, CV_8UC1);

	cv::threshold(confidentMat, *m_pConfidentMask, FINAL_CONFIDENT_MEASURE_THRES, 255, THRESH_BINARY);
	confidentMaskRepair(*m_pConfidentMask, rawImageParameter, microImageParameter);
	cv::imwrite(confidentMaskName, *m_pConfidentMask);

	std::string storeName = m_folderPath + "/confidentMatMask.xml";
	storeDispMapToXML(storeName, *m_pConfidentMask);
}

void ConfidenceCompute::confidentCircleJudge(cv::Mat &confidentMat, cv::Mat &mask, const RawImageParameter &rawImageParameter,
	const MicroImageParameter &microImageParameter)
{
//#pragma omp parallel for
	for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
	{
		uchar *yRows = (uchar *)mask.ptr<uchar>(y);
		for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
		{
			if (!confidentCircleJudge(confidentMat, y, x, rawImageParameter, microImageParameter))
				yRows[x] = 255;
		}
	}
}

bool ConfidenceCompute::confidentCircleJudge(cv::Mat &confidentMat, int y, int x, const RawImageParameter &rawImageParameter,
	const MicroImageParameter &microImageParameter)
{
	Point2d &centerPos = microImageParameter.m_ppLensCenterPoints[y][x];
	int curCenterIndex = y*rawImageParameter.m_xLensNum + x;
	double sumCircle = 0.0, countPoints = 0;

	for (int py = centerPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
		py <= centerPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; py++)
	{
		for (int px = centerPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
			px <= centerPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; px++)
		{
			uchar *yConf = confidentMat.ptr<uchar>(py - rawImageParameter.m_yPixelBeginOffset);

			if (microImageParameter.m_ppPixelsMappingSet[py][px] == curCenterIndex){
				sumCircle += yConf[px - rawImageParameter.m_xPixelBeginOffset];
				countPoints += 1.0;
			}
		}
	}

	if (sumCircle / countPoints > MNN_CONFIDENT_MEASURE_THRES)
		return true;
	else
		return false;
}

void ConfidenceCompute::drawConfidentCircle(cv::Mat *&costVol, cv::Mat &mask, const RawImageParameter &rawImageParameter,
	const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter, cv::Mat &srcImg)
{
	cv::Mat rawDisp = cv::Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_8UC1);
	WTAMatch(costVol, rawDisp, disparityParameter.m_disNum);
	double minVal, maxVal;
	minMaxLoc(rawDisp, &minVal, &maxVal);
	cv::Mat disp2;
	rawDisp.convertTo(disp2, CV_8UC1, 255.0 / (maxVal - minVal), -minVal*255.0 / (maxVal - minVal));

	cv::Mat inputImg;
	srcImg.convertTo(inputImg, CV_8UC3);
	//cv::cvtColor(inputImg, inputImg, CV_RGB2BGR);

	std::string rawImgStoreName = m_folderPath + "/ori_circle.png";
	lowTextureAreaPlot(rawImageParameter, microImageParameter, inputImg, mask, rawImgStoreName, CircleDrawMode::e_color, true);

	std::string dispImgStoreName = m_folderPath + "/disp_circle.png";
	lowTextureAreaPlot(rawImageParameter, microImageParameter, disp2, mask, dispImgStoreName, CircleDrawMode::e_gray, true);
}

void ConfidenceCompute::lowTextureAreaPlot(const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter,
	const Mat &img_input, const Mat &mask, const string &picName, CircleDrawMode _circleDrawMode, bool _isOffset)
{
	Mat res_input = img_input.clone();
	int offsetY = _isOffset ? rawImageParameter.m_yPixelBeginOffset : 0;
	int offsetX = _isOffset ? rawImageParameter.m_xPixelBeginOffset : 0;
//#pragma omp parallel for
	for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
	{
		uchar *yRows = (uchar *)mask.ptr<uchar>(y);
		for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
		{
			if (yRows[x] > 125)
			{
				Point2d curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x] - Point2d(offsetX, offsetY);
				drawCircle(res_input, curCenterPos, _circleDrawMode);
			}
		}
	}
	imwrite(picName, res_input);
}

void ConfidenceCompute::drawCircle(Mat &img, const Point2d &centerPos, CircleDrawMode _circleDrawMode)
{
	for (double angle = 0; angle < 360.0; angle += 1.0)
	{
		int index_y = int(centerPos.y - RADIUS*sin(angle / 180.0*PI) + 0.5);
		int index_x = int(centerPos.x + RADIUS*cos(angle / 180.0*PI) + 0.5);

		uchar *ydata = (uchar *)img.ptr<uchar>(index_y);
		if (_circleDrawMode == CircleDrawMode::e_color)
		{
			ydata[3 * index_x] = 0; ydata[3 * index_x + 1] = 0; ydata[3 * index_x + 2] = 255;
		}
		else if (_circleDrawMode == CircleDrawMode::e_gray)
			ydata[index_x] = 255;
	}
}

void ConfidenceCompute::confidentMaskRepair(cv::Mat &confidentMat, const RawImageParameter &rawImageParameter,
	const MicroImageParameter &microImageParameter)
{//置信度图利用梯度圆做修复
	int height = confidentMat.rows;
	int width = confidentMat.cols;

//#pragma omp parallel for
	for (int py = 0; py < height; ++py)
	{
		uchar *pYMask = (uchar *)confidentMat.ptr<uchar>(py);
		for (int px = 0; px < width; ++px)
		{
			int c_index = microImageParameter.m_ppPixelsMappingSet[py][px];
			if (c_index > 0){
				int cy = c_index / rawImageParameter.m_xLensNum;
				int cx = c_index % rawImageParameter.m_xLensNum;

				uchar *ydata = (uchar *)(*m_pGradientCircleMask).ptr<uchar>(cy);
				if (ydata[cx] == 0)
					pYMask[px] = 0;
				//if (m_pGradientCircleMask->at<uchar>(cy, cx) == 0)
					//pYMask[px] = 0;
			}
		}
	}
}