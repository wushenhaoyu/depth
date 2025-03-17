#include "CostVolFilter.h"
#include "DataParameter.h"
#include "DisparityRefinement.h"
//#include "opencv2/gpu/gpu.hpp"  
#include <time.h>
using namespace cv;
using namespace std;

void CostVolFilter::costVolWindowFilter(const DataParameter &dataParameter, cv::Mat *costVol, cv::Mat *costVolFilter)
{//��costVol���д����˲�
	RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
	MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
	DisparityParameter disparityParameter = dataParameter.getDisparityParameter();
	FilterPatameter filterPatameter = dataParameter.getFilterPatameter();

//#pragma omp parallel for
	for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
	{
		for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
		{
			//std::cout << "cost filter --- y=" << y << "\tx=" << x << std::endl;
			costVolWindowFilter(costVol, costVolFilter, y, x, rawImageParameter, microImageParameter, disparityParameter, filterPatameter);
		}
	}

	//costVolBoundaryRepair(costVolFilter, disparityParameter, filterPatameter);
}

void CostVolFilter::costVolWindowFilter(cv::Mat *costVol, cv::Mat *costVolFilter, int y, int x, const RawImageParameter &rawImageParameter,
	const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter, const FilterPatameter &filterPatameter)
{//��costVol���д����˲�
	Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];
	int xBegin = curCenterPos.x  - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
	int yBegin = curCenterPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
	int xEnd = curCenterPos.x  + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow;
	int yEnd = curCenterPos.y  + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow;

	cv::Mat divideMask = (*filterPatameter.m_pValidNeighborPixelsNum)(cv::Rect(xBegin, yBegin, xEnd - xBegin + 1, yEnd - yBegin + 1));
	cv::Mat multiMask = (*filterPatameter.m_pValidPixelsMask)(cv::Rect(xBegin, yBegin, xEnd - xBegin + 1, yEnd - yBegin + 1));

	for (int d = 0; d < disparityParameter.m_disNum; d++)
	{
		cv::Mat srcCost = costVol[d](cv::Rect(xBegin - rawImageParameter.m_xPixelBeginOffset, yBegin - rawImageParameter.m_yPixelBeginOffset, xEnd - xBegin + 1, yEnd - yBegin + 1));
		cv::Mat destCost = costVolFilter[d](cv::Rect(xBegin - rawImageParameter.m_xPixelBeginOffset, yBegin - rawImageParameter.m_yPixelBeginOffset, xEnd - xBegin + 1, yEnd - yBegin + 1));
		//cv::blur(srcCost, destCost, cv::Size(m_filter_radius, m_filter_radius));
		cv::filter2D(srcCost, destCost, -1, filterPatameter.m_filterKnernel, cv::Point(-1, -1), 0, BORDER_CONSTANT);
		cv::divide(destCost, divideMask, destCost);
		cv::multiply(destCost, multiMask, destCost);
	}
}

void CostVolFilter::costVolBoundaryRepair(cv::Mat *costVol, const DisparityParameter &disparityParameter, const FilterPatameter &filterPatameter)
{//��Բ�α߽���д���
//#pragma omp parallel for
	for (int d = 0; d < disparityParameter.m_disNum; d++)
	{
		costVol[d] = costVol[d].mul(*filterPatameter.m_pValidPixelsMask);
		//std::cout << "cost repair --- d=" << d << std::endl;
	}
}

/*void CostVolFilter::microImageDisparityFilter(const DataParameter &dataParameter, cv::Mat *&costVol, FilterOptimizeKind curFilterOptimizeKind)
{//��С��microͼ������˲�
	RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
	MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
	DisparityParameter disparityParameter = dataParameter.getDisparityParameter();
	FilterPatameter filterPatameter = dataParameter.getFilterPatameter();
	cv::Mat inputImg;
	dataParameter.m_inputImgRec.convertTo(inputImg, CV_32FC3, 1 / 255.0f);

#pragma omp parallel for
	for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
	{
		for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
		{
			//std::cout << "micro --- y=" << y << "\tx=" << x << std::endl;
			Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];

			int xBegin = curCenterPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow - rawImageParameter.m_xPixelBeginOffset;
			int yBegin = curCenterPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow - rawImageParameter.m_yPixelBeginOffset;
			int xEnd = curCenterPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow - rawImageParameter.m_xPixelBeginOffset;
			int yEnd = curCenterPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow - rawImageParameter.m_yPixelBeginOffset;

			cv::Mat referImg = inputImg(cv::Rect(xBegin, yBegin, xEnd - xBegin + 1, yEnd - yBegin + 1));
			//cv::Mat multiMask = (*filterPatameter.m_pValidPixelsMask)(cv::Rect(xBegin, yBegin, xEnd - xBegin + 1, yEnd - yBegin + 1));

			cv::Mat *costFilter = new cv::Mat[disparityParameter.m_disNum];
			for (int d = 0; d < disparityParameter.m_disNum; d++) 
				costFilter[d] = costVol[d](cv::Rect(xBegin, yBegin, xEnd - xBegin + 1, yEnd - yBegin + 1));
			
			DisparityRefinement::getInstance()->localFilterOrOptimize(referImg, referImg, disparityParameter.m_disNum, costFilter, curFilterOptimizeKind);

			delete[]costFilter;
		}
	}
}*/


__global__ void costVolWindowFilterKernel(
    float* costVol, float* divideMask, float* multiMask,
    float* filterKernel, int width, int height, int channels,
    int kernelSize, int xBegin, int yBegin, int xEnd, int yEnd,
    int xPixelBeginOffset, int yPixelBeginOffset, int disNum) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查是否在处理区域内
    if (x < xBegin || x > xEnd || y < yBegin || y > yEnd) {
        return;
    }

    // 计算当前像素的全局索引
    int globalIdx = (y * width + x) * channels;

    // 遍历每个视差值
    for (int d = 0; d < disNum; d++) {
        int srcIdx = ((y - yPixelBeginOffset) * width + (x - xPixelBeginOffset)) * channels + d;
        float sum = 0.0f;

        // 应用卷积核
        for (int ky = -kernelSize / 2; ky <= kernelSize / 2; ky++) {
            for (int kx = -kernelSize / 2; kx <= kernelSize / 2; kx++) {
                int nx = x + kx;
                int ny = y + ky;

                // 检查边界
                if (nx >= xBegin && nx <= xEnd && ny >= yBegin && ny <= yEnd) {
                    int neighborIdx = ((ny - yPixelBeginOffset) * width + (nx - xPixelBeginOffset)) * channels + d;
                    int kernelIdx = (ky + kernelSize / 2) * kernelSize + (kx + kernelSize / 2);
                    sum += costVol[neighborIdx] * filterKernel[kernelIdx];
                }
            }
        }

        // 应用除法和乘法
        float divideValue = divideMask[globalIdx];
        float multiValue = multiMask[globalIdx];

        if (divideValue != 0.0f) {
            sum /= divideValue;
            sum *= multiValue;
        }

        // 将结果写回
        costVol[srcIdx] = sum;
    }
}


void CostVolFilter::costVolWindowFilter(const DataParameter &dataParameter, cv::Mat *costVol)
{
    // 获取参数
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();
    FilterPatameter filterPatameter = dataParameter.getFilterPatameter();

    int width = rawImageParameter.m_xLensNum;
    int height = rawImageParameter.m_yLensNum;
    int channels = disparityParameter.m_disNum;
    int kernelSize = filterPatameter.m_filterKnernel.rows;

    // 分配 GPU 内存
    float* d_costVol;
    float* d_divideMask;
    float* d_multiMask;
    float* d_filterKernel;

    cudaMalloc(&d_costVol, width * height * channels * sizeof(float));
    cudaMalloc(&d_divideMask, width * height * sizeof(float));
    cudaMalloc(&d_multiMask, width * height * sizeof(float));
    cudaMalloc(&d_filterKernel, kernelSize * kernelSize * sizeof(float));

    // 数据传输到 GPU
    cudaMemcpy(d_costVol, costVol[0].ptr<float>(), width * height * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_divideMask, filterPatameter.m_pValidNeighborPixelsNum->ptr<float>(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_multiMask, filterPatameter.m_pValidPixelsMask->ptr<float>(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filterKernel, filterPatameter.m_filterKnernel.ptr<float>(), kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块大小和网格大小
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 调用核函数
    costVolWindowFilterKernel<<<gridSize, blockSize>>>(
        d_costVol, d_divideMask, d_multiMask, d_filterKernel,
        width, height, channels, kernelSize,
        microImageParameter.m_circleNarrow, microImageParameter.m_circleNarrow,
        width - microImageParameter.m_circleNarrow, height - microImageParameter.m_circleNarrow,
        rawImageParameter.m_xPixelBeginOffset, rawImageParameter.m_yPixelBeginOffset,
        disparityParameter.m_disNum
    );

    // 同步检查
    cudaDeviceSynchronize();

    // 将结果传输回 CPU
    cudaMemcpy(costVol[0].ptr<float>(), d_costVol, width * height * channels * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_costVol);
    cudaFree(d_divideMask);
    cudaFree(d_multiMask);
    cudaFree(d_filterKernel);
}