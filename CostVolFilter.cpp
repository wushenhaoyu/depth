#include "CostVolFilter.h"
#include "DataParameter.h"
#include "DisparityRefinement.h"
//#include "opencv2/gpu/gpu.hpp"  
#include <time.h>
using namespace cv;
using namespace std;

void CostVolFilter::costVolWindowFilter(const DataParameter &dataParameter, cv::Mat *costVol, cv::Mat *costVolFilter)
{//锟斤拷costVol锟斤拷锟叫达拷锟斤拷锟剿诧拷
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
{//锟斤拷costVol锟斤拷锟叫达拷锟斤拷锟剿诧拷
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
{//锟斤拷圆锟轿边斤拷锟斤拷写锟斤拷锟�
//#pragma omp parallel for
	for (int d = 0; d < disparityParameter.m_disNum; d++)
	{
		costVol[d] = costVol[d].mul(*filterPatameter.m_pValidPixelsMask);
		//std::cout << "cost repair --- d=" << d << std::endl;
	}
}

/*void CostVolFilter::microImageDisparityFilter(const DataParameter &dataParameter, cv::Mat *&costVol, FilterOptimizeKind curFilterOptimizeKind)
{//锟斤拷小锟斤拷micro图锟斤拷锟斤拷锟斤拷瞬锟�
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
#include <opencv2/cudafilters.hpp>  // CUDA 滤波器模块
#include <opencv2/cudaarithm.hpp>   // CUDA 算术操作模块
#include <omp.h>                    // OpenMP 并行化

void CostVolFilter::costVolWindowFilter(const DataParameter &dataParameter, cv::Mat *costVol) {
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();
    FilterPatameter filterPatameter = dataParameter.getFilterPatameter();

    // 使用 OpenMP 并行化外层循环
    #pragma omp parallel for
    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++) {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++) {
            costVolWindowFilter(costVol, y, x, rawImageParameter, microImageParameter, disparityParameter, filterPatameter);
        }
    }
}

void CostVolFilter::costVolWindowFilter(cv::Mat *costVol, int y, int x, const RawImageParameter &rawImageParameter,
    const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter, const FilterPatameter &filterPatameter) {
    Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];
    int xBegin = curCenterPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
    int yBegin = curCenterPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
    int xEnd = curCenterPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow;
    int yEnd = curCenterPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow;

    // 检查边界，确保矩形区域在图像范围内
    xBegin = max(0, xBegin);
    yBegin = max(0, yBegin);
    xEnd = min(filterPatameter.m_pValidNeighborPixelsNum->cols - 1, xEnd);
    yEnd = min(filterPatameter.m_pValidNeighborPixelsNum->rows - 1, yEnd);

    if (xEnd < xBegin || yEnd < yBegin) {
        return; // 如果区域无效，直接返回
    }

    // 提取 divideMask 和 multiMask
    cv::Mat divideMask = (*filterPatameter.m_pValidNeighborPixelsNum)(cv::Rect(xBegin, yBegin, xEnd - xBegin + 1, yEnd - yBegin + 1));
    cv::Mat multiMask = (*filterPatameter.m_pValidPixelsMask)(cv::Rect(xBegin, yBegin, xEnd - xBegin + 1, yEnd - yBegin + 1));

    // 使用 CUDA 加速滤波操作
    cv::cuda::GpuMat gpuDivideMask, gpuMultiMask, gpuSrcCost, gpuDestCost;
    gpuDivideMask.upload(divideMask);
    gpuMultiMask.upload(multiMask);

    // 创建 CUDA 滤波器
    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createLinearFilter(
        CV_32F, CV_32F, filterPatameter.m_filterKnernel, cv::Point(-1, -1), cv::BORDER_CONSTANT);

    for (int d = 0; d < disparityParameter.m_disNum; d++) {
        // 提取 srcCost
        cv::Mat srcCost = costVol[d](cv::Rect(xBegin - rawImageParameter.m_xPixelBeginOffset, yBegin - rawImageParameter.m_yPixelBeginOffset, xEnd - xBegin + 1, yEnd - yBegin + 1));
        gpuSrcCost.upload(srcCost);

        // 使用 CUDA 进行滤波
        filter->apply(gpuSrcCost, gpuDestCost);

        // 使用 CUDA 进行除法和乘法操作
        cv::cuda::divide(gpuDestCost, gpuDivideMask, gpuDestCost);
        cv::cuda::multiply(gpuDestCost, gpuMultiMask, gpuDestCost);

        // 下载结果到 CPU
        gpuDestCost.download(srcCost);
    }
}

/*
void CostVolFilter::gpuFilterTest(const DataParameter &dataParameter, cv::Mat *&costVol)
{
	RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
	MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
	DisparityParameter disparityParameter = dataParameter.getDisparityParameter();
	FilterPatameter filterPatameter = dataParameter.getFilterPatameter();

	cv::Mat divideMask_host = (*filterPatameter.m_pValidNeighborPixelsNum)(cv::Rect(rawImageParameter.m_xPixelBeginOffset, 
		rawImageParameter.m_yPixelBeginOffset, rawImageParameter.m_recImgWidth, rawImageParameter.m_recImgHeight));
	cv::Mat multiMask_host = (*filterPatameter.m_pValidPixelsMask)(cv::Rect(rawImageParameter.m_xPixelBeginOffset,
		rawImageParameter.m_yPixelBeginOffset, rawImageParameter.m_recImgWidth, rawImageParameter.m_recImgHeight));
	
	cv::gpu::setDevice(0);
	cv::gpu::GpuMat divideMask_gpu(divideMask_host);
	cv::gpu::GpuMat multiMask_gpu(multiMask_host);
	cv::gpu::GpuMat cost_gpu, cost_gpu_temp;

	clock_t t1 = clock();
	for (int d = 0; d < disparityParameter.m_disNum; d++)
	{
		cost_gpu.upload(costVol[d]);
		cv::gpu::filter2D(cost_gpu, cost_gpu_temp, -1, filterPatameter.m_filterKnernel, cv::Point(-1, -1), 0);
		cv::gpu::divide(cost_gpu_temp, divideMask_gpu, cost_gpu_temp);
		cv::gpu::multiply(cost_gpu_temp, multiMask_gpu, cost_gpu_temp);
		cost_gpu_temp.download(costVol[d]);
	}
	clock_t t2 = clock();
	std::cout << "gpu real filter use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds " << std::endl;
}
*/