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


extern __constant__ RawImageParameter d_rawImageParameter;
extern __constant__ DisparityParameter d_disparityParameter;
extern __constant__ FilterParameterDevice d_filterPatameterDevice; 
extern __device__ MicroImageParameterDevice d_microImageParameter; 
extern __device__ float* d_costVol;

__global__ void costVolWindowFilterKernel()
{
    // 获取线程的坐标
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保线程不超出图像范围
    if (y < d_rawImageParameter.m_yCenterBeginOffset || 
        y >= d_rawImageParameter.m_yLensNum - d_rawImageParameter.m_yCenterEndOffset ||
        x < d_rawImageParameter.m_xCenterBeginOffset ||
        x >= d_rawImageParameter.m_xLensNum - d_rawImageParameter.m_xCenterEndOffset)
    {
        return; // 如果超出范围，跳过该线程
    }

    // 加载常用数据到共享内存
    extern __shared__ float sharedMem[];
    int* validNeighborPixelsNum = (int*)sharedMem;
    int* validPixelsMask = (int*)&validNeighborPixelsNum[blockDim.x * blockDim.y]; 
    float* filterKernel = (float*)&validPixelsMask[blockDim.x * blockDim.y];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // 复制相关数据到共享内存
    if (idx < blockDim.x * blockDim.y) {
        validNeighborPixelsNum[idx] = d_filterPatameterDevice.d_validNeighborPixelsNum[y * d_rawImageParameter.m_xLensNum + x];
        validPixelsMask[idx] = d_filterPatameterDevice.d_validPixelsMask[y * d_rawImageParameter.m_xLensNum + x];
        filterKernel[idx] = d_filterPatameterDevice.d_filterKernel[idx];
    }

    __syncthreads();  // 确保数据加载完毕

    // 获取当前中心点位置
    Point2d curCenterPos = d_microImageParameter.d_lensCenterPoints[y * d_rawImageParameter.m_xLensNum + x];
    int xBegin = curCenterPos.x - d_microImageParameter.d_circleDiameter / 2 + d_microImageParameter.d_circleNarrow;
    int yBegin = curCenterPos.y - d_microImageParameter.d_circleDiameter / 2 + d_microImageParameter.d_circleNarrow;
    int xEnd = curCenterPos.x + d_microImageParameter.d_circleDiameter / 2 - d_microImageParameter.d_circleNarrow;
    int yEnd = curCenterPos.y + d_microImageParameter.d_circleDiameter / 2 - d_microImageParameter.d_circleNarrow;

    // 获取掩膜值
    int maskWidth = xEnd - xBegin + 1;
    int maskHeight = yEnd - yBegin + 1;

    // 对每个视差进行卷积操作
    for (int d = 0; d < d_disparityParameter.m_disNum; d++) {
        // 计算代价体中的区域
        float* srcCost = &d_costVol[d * d_rawImageParameter.m_recImgHeight * d_rawImageParameter.m_recImgWidth + y * d_rawImageParameter.m_recImgWidth + x];

        // 做卷积操作：filter2D
        float filteredValue = 0.0f;
        for (int dy = -maskHeight / 2; dy <= maskHeight / 2; dy++) {
            for (int dx = -maskWidth / 2; dx <= maskWidth / 2; dx++) {
                int px = x + dx;
                int py = y + dy;
                if (px >= 0 && px < d_rawImageParameter.m_recImgWidth && py >= 0 && py < d_rawImageParameter.m_recImgHeight) {
                    // 使用filterKernel进行卷积操作
                    float weight = filterKernel[(dy + maskHeight / 2) * maskWidth + (dx + maskWidth / 2)];
                    filteredValue += weight * srcCost[py * d_rawImageParameter.m_recImgWidth + px];
                }
            }
        }

        // 用divideMask和multiMask进行归一化和过滤
        filteredValue /= validNeighborPixelsNum[0];  // 由于只需一个有效的值
        filteredValue *= validPixelsMask[0];

        // 更新代价体
        srcCost[0] = filteredValue;
    }
}

void CostVolFilter::costVolWindowFilter(const DataParameter &dataParameter, cv::Mat *costVol)
{
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
     // 启动 CUDA 核函数
     dim3 blockDim(32, 32);  
     dim3 gridDim((rawImageParameter.m_xLensNum + blockDim.x - 1) / blockDim.x, 
                  (rawImageParameter.m_yLensNum + blockDim.y - 1) / blockDim.y);
 
     // 启动 CUDA 核函数
     costVolWindowFilterKernel<<<gridDim, blockDim>>>();
 
     // 检查 CUDA 错误
     cudaDeviceSynchronize();
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
     }
}