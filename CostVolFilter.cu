#include "CostVolFilter.h"
#include "DataParameter.cuh"
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

/*__global__ void testKernel() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // 在一个特定的索引位置修改 d_ppLensMeanDisp
   // printf("m_xLensNum = %d, m_yLensNum = %d\n", d_rawImageParameter.m_xLensNum, d_rawImageParameter.m_yLensNum);
    
    
    if (x < d_rawImageParameter.m_xLensNum && y < d_rawImageParameter.m_yLensNum) {
        int index = y * d_rawImageParameter.m_xLensNum + x;
        if (index == 0) {  
            d_ppLensMeanDisp[index] = 42.0f;  
        }
    }
}*/


__global__ void costVolWindowFilterKernel()
{

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < d_rawImageParameter.m_yCenterBeginOffset || 
        y >= d_rawImageParameter.m_yLensNum - d_rawImageParameter.m_yCenterEndOffset ||
        x < d_rawImageParameter.m_xCenterBeginOffset ||
        x >= d_rawImageParameter.m_xLensNum - d_rawImageParameter.m_xCenterEndOffset)
    {
        return;
    }


    Point2d curCenterPos = d_microImageParameter.m_ppLensCenterPoints[y * d_rawImageParameter.m_xLensNum + x];
    int xBegin = curCenterPos.x - d_microImageParameter.m_circleDiameter / 2 + d_microImageParameter.m_circleNarrow;
    int yBegin = curCenterPos.y - d_microImageParameter.m_circleDiameter / 2 + d_microImageParameter.m_circleNarrow;
    int xEnd = curCenterPos.x + d_microImageParameter.m_circleDiameter / 2 - d_microImageParameter.m_circleNarrow;
    int yEnd = curCenterPos.y + d_microImageParameter.m_circleDiameter / 2 - d_microImageParameter.m_circleNarrow;


    int maskWidth = xEnd - xBegin + 1;
    int maskHeight = yEnd - yBegin + 1;

    for (int d = 0; d < d_disparityParameter.m_disNum; d++) {
        //printf("d = %d, x = %d , y = %d \n", d,x,y);
        float* srcCost = &d_costVol[d * d_rawImageParameter.m_recImgHeight * d_rawImageParameter.m_recImgWidth + y * d_rawImageParameter.m_recImgWidth + x];


        float filteredValue = 0.0f;
        //printf("filteredValue = %f\n", filteredValue);
        for (int dy = -maskHeight / 2; dy <= maskHeight / 2; dy++) {
            for (int dx = -maskWidth / 2; dx <= maskWidth / 2; dx++) {
                int px = x + dx;
                int py = y + dy;
                if (px >= 0 && px < d_rawImageParameter.m_recImgWidth && py >= 0 && py < d_rawImageParameter.m_recImgHeight) {
                    float weight = d_filterPatameterDevice.d_filterKernel[(dy + maskHeight / 2) * maskWidth + (dx + maskWidth / 2)];
                    filteredValue += weight * srcCost[py * d_rawImageParameter.m_recImgWidth + px];
                }
            }
        }


        filteredValue /= d_filterPatameterDevice.d_validNeighborPixelsNum[0];  
        filteredValue *= d_filterPatameterDevice.d_validPixelsMask[0];
        srcCost[0] = filteredValue;
        //printf("filteredValue = %f\n", srcCost[0]);

    }
}

void CostVolFilter::costVolWindowFilter(const DataParameter &dataParameter, cv::Mat *costVol)
{
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();

    // 启动 CUDA 核函数
    dim3 blockDim(32, 32);  
    dim3 gridDim((rawImageParameter.m_xLensNum + blockDim.x - 1) / blockDim.x, 
                 (rawImageParameter.m_yLensNum + blockDim.y - 1) / blockDim.y);

    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 启动 CUDA 核函数
    cudaEventRecord(start); // 记录开始时间
    costVolWindowFilterKernel<<<gridDim, blockDim>>>();
    cudaEventRecord(stop);  // 记录结束时间

    // 检查 CUDA 错误
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计算运行时间
    float milliseconds = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 打印运行时间
    std::cout << "CUDA kernel execution time: " << milliseconds << " ms" << std::endl;
    if (d_ppLensMeanDisp == nullptr) {
        printf("d_ppLensMeanDisp is NULL\n");
    }
    else{
        printf("d_ppLensMeanDisp is not NULL\n");
    }
    // 销毁 CUDA 事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //testKernel<<<gridDim, blockDim>>>();
    //CUDA_CHECK(cudaGetLastError());
    //CUDA_CHECK(cudaDeviceSynchronize());
}


/*
void CostVolFilter::costVolWindowFilter(const DataParameter &dataParameter, cv::Mat *costVol)
{
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();
    FilterPatameter filterPatameter = dataParameter.getFilterPatameter();

    // Loop over each pixel in the cost volume
    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
    {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
        {
            // Get the current lens center position for this pixel
            Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];
            int xBegin = curCenterPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
            int yBegin = curCenterPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow;
            int xEnd = curCenterPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow;
            int yEnd = curCenterPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow;

            // Extract relevant masks for filtering
            cv::Mat divideMask = (*filterPatameter.m_pValidNeighborPixelsNum)(cv::Rect(xBegin, yBegin, xEnd - xBegin + 1, yEnd - yBegin + 1));
            cv::Mat multiMask = (*filterPatameter.m_pValidPixelsMask)(cv::Rect(xBegin, yBegin, xEnd - xBegin + 1, yEnd - yBegin + 1));
            cv::Mat destCost;

            // Iterate over each disparity level and apply the filtering process
            for (int d = 0; d < disparityParameter.m_disNum; d++)
            {
                // Extract the source cost matrix for the current disparity level
                cv::Mat srcCost = costVol[d](cv::Rect(xBegin - rawImageParameter.m_xPixelBeginOffset, 
                                                      yBegin - rawImageParameter.m_yPixelBeginOffset, 
                                                      xEnd - xBegin + 1, 
                                                      yEnd - yBegin + 1));
                // Apply the filter to the source cost matrix
                cv::filter2D(srcCost, destCost, -1, filterPatameter.m_filterKnernel, cv::Point(-1, -1), 0, BORDER_CONSTANT);
                cv::divide(destCost, divideMask, destCost);  // Divide by the valid neighbor pixels mask
                cv::multiply(destCost, multiMask, destCost);  // Multiply by the valid pixels mask
                destCost.copyTo(srcCost);  // Copy the result back to the source cost matrix
            }
        }
    }
}
*/