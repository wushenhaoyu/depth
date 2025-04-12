#include "CostVolFilter.h"
#include "DataParameter.cuh"
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

__global__ void costVolWindowFilterKernel(MicroImageParameterDevice* d_microImageParameter,FilterParameterDevice* d_filterPatameterDevice
) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < d_rawImageParameter.m_yCenterBeginOffset || 
        y >= d_rawImageParameter.m_yLensNum - d_rawImageParameter.m_yCenterEndOffset ||
        x < d_rawImageParameter.m_xCenterBeginOffset ||
        x >= d_rawImageParameter.m_xLensNum - d_rawImageParameter.m_xCenterEndOffset)
    {
        return;
    }

    CudaPoint2f curCenterPos = d_microImageParameter->m_ppLensCenterPoints[y * d_rawImageParameter.m_xLensNum + x];
    //printf("x:%d y:%d sx:%d sy:%d\n",x,y,curCenterPos.x,curCenterPos.y);
    int xBegin = curCenterPos.x - d_microImageParameter->m_circleDiameter / 2 + d_microImageParameter->m_circleNarrow;
    int yBegin = curCenterPos.y - d_microImageParameter->m_circleDiameter / 2 + d_microImageParameter->m_circleNarrow;
    int xEnd = curCenterPos.x + d_microImageParameter->m_circleDiameter / 2 - d_microImageParameter->m_circleNarrow;
    int yEnd = curCenterPos.y + d_microImageParameter->m_circleDiameter / 2 - d_microImageParameter->m_circleNarrow;
    int maskWidth = xEnd - xBegin + 1;
    int maskHeight = yEnd - yBegin + 1;
    for (int d = 0; d < d_disparityParameter.m_disNum; d++) {
        float destCost[38*38]; //Cuda不支持动态数组,这里写死了

        
    for (int localY = 0; localY < maskHeight; localY++) {
        for (int localX = 0; localX < maskWidth; localX++) {
            float filteredValue = 0.0f;
            int globalX = xBegin + localX;
            int globalY = yBegin + localY;
            if (globalX >= 0 && globalX < d_rawImageParameter.m_recImgWidth &&
                globalY >= 0 && globalY < d_rawImageParameter.m_recImgHeight) {
                    for (int dy = -d_filterRadius; dy <= d_filterRadius; dy++) {
                        for (int dx = -d_filterRadius; dx <= d_filterRadius; dx++) {
                            int globalPx = globalX + dx;
                            int globalPy = globalY + dy;
                            int localPx = localX + dx;
                            int localPy = localY + dy;
                            if (localPx >= 0 && localPx < maskWidth && localPy >= 0 && localPy < maskHeight) {
                                filteredValue += d_costVol[d * d_rawImageParameter.m_recImgHeight * d_rawImageParameter.m_recImgWidth + (globalPy -  d_rawImageParameter.m_yPixelBeginOffset)  * d_rawImageParameter.m_recImgWidth + (globalPx -  d_rawImageParameter.m_xPixelBeginOffset)];
                            }
                        }
                    }
                    
                    float* src = d_costVol + d * d_rawImageParameter.m_recImgHeight * d_rawImageParameter.m_recImgWidth + (globalY - d_rawImageParameter.m_yPixelBeginOffset)* d_rawImageParameter.m_recImgWidth + globalX - d_rawImageParameter.m_xPixelBeginOffset;
                    float divide = d_filterPatameterDevice->d_validNeighborPixelsNum[globalY* d_rawImageParameter.m_srcImgWidth+ globalX];
                    float multiply = d_filterPatameterDevice->d_validPixelsMask[globalY*d_rawImageParameter.m_srcImgWidth + globalX];

                   /* if(x == 2 && y == 2 && d == 35)
                    {
                       printf("i:%d,j:%d,divide:%f multiply:%f,res:%f,src:%f\n",localY,localX,divide,multiply,filteredValue,destCost[localY * maskWidth + localX],src[0]);
                    }*/
                    destCost[localY * maskWidth + localX] = filteredValue / divide * multiply;
            } 
        }
    }

    for (int localY = 0; localY < maskHeight; localY++) { 
        for (int localX = 0; localX < maskWidth; localX++) {
            int globalX = xBegin + localX;
            int globalY = yBegin + localY;
            d_costVol[d * d_rawImageParameter.m_recImgHeight * d_rawImageParameter.m_recImgWidth + (globalY - d_rawImageParameter.m_yPixelBeginOffset)* d_rawImageParameter.m_recImgWidth + globalX - d_rawImageParameter.m_xPixelBeginOffset] = destCost[localY * maskWidth + localX];
        }
    }
    }
}

void CostVolFilter::costVolWindowFilter(const DataParameter &dataParameter, cv::Mat *costVol) {
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();
    FilterPatameter filterPatameter = dataParameter.getFilterPatameter();


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
    costVolWindowFilterKernel<<<gridDim, blockDim>>>(d_microImageParameter,d_filterPatameterDevice);
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

    // 销毁 CUDA 事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}