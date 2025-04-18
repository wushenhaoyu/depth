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


__global__ void costVolWindowFilterKernel(
    MicroImageParameterDevice* d_microImageParameter,
    FilterParameterDevice* d_filterPatameterDevice,
    int disparityNum)
{
    int patchX = blockIdx.x;
    int patchY = blockIdx.y;

    if (patchX < d_rawImageParameter.m_xCenterBeginOffset || 
        patchX >= d_rawImageParameter.m_xLensNum - d_rawImageParameter.m_xCenterEndOffset ||
        patchY < d_rawImageParameter.m_yCenterBeginOffset || 
        patchY >= d_rawImageParameter.m_yLensNum - d_rawImageParameter.m_yCenterEndOffset)
        return;

    CudaPoint2f curCenterPos = d_microImageParameter->m_ppLensCenterPoints[patchY * d_rawImageParameter.m_xLensNum + patchX];

    int xBegin = curCenterPos.x - d_microImageParameter->m_circleDiameter / 2 + d_microImageParameter->m_circleNarrow;
    int yBegin = curCenterPos.y - d_microImageParameter->m_circleDiameter / 2 + d_microImageParameter->m_circleNarrow;
    int xEnd   = curCenterPos.x + d_microImageParameter->m_circleDiameter / 2 - d_microImageParameter->m_circleNarrow;
    int yEnd   = curCenterPos.y + d_microImageParameter->m_circleDiameter / 2 - d_microImageParameter->m_circleNarrow;

    int maskWidth  = xEnd - xBegin + 1;
    int maskHeight = yEnd - yBegin + 1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int d  = threadIdx.z;

    int stride_x = blockDim.x;
    int stride_y = blockDim.y;
    int stride_d = blockDim.z;

    for (int localY = ty; localY < maskHeight; localY += stride_y) {
        for (int localX = tx; localX < maskWidth; localX += stride_x) {
            for (int disp = d; disp < d_disparityParameter.m_disNum; disp += stride_d)
            {
                int globalX = xBegin + localX;
                int globalY = yBegin + localY;

                if (globalX < 0 || globalX >= d_rawImageParameter.m_recImgWidth ||
                    globalY < 0 || globalY >= d_rawImageParameter.m_recImgHeight)
                    continue;

                float filteredValue = 0.0f;

                for (int dy = -d_filterRadius; dy <= d_filterRadius; ++dy) {
                    for (int dx = -d_filterRadius; dx <= d_filterRadius; ++dx) {
                        int px = globalX + dx;
                        int py = globalY + dy;

                        if (px < 0 || px >= d_rawImageParameter.m_recImgWidth ||
                            py < 0 || py >= d_rawImageParameter.m_recImgHeight)
                            continue;

                        int costIdx = disp * d_rawImageParameter.m_recImgHeight * d_rawImageParameter.m_recImgWidth +
                                      (py - d_rawImageParameter.m_yPixelBeginOffset) * d_rawImageParameter.m_recImgWidth +
                                      (px - d_rawImageParameter.m_xPixelBeginOffset);

                        filteredValue += d_costVol[costIdx];
                    }
                }

                float divide = d_filterPatameterDevice->d_validNeighborPixelsNum[globalY * d_rawImageParameter.m_srcImgWidth + globalX];
                float multiply = d_filterPatameterDevice->d_validPixelsMask[globalY * d_rawImageParameter.m_srcImgWidth + globalX];

                int writeX = globalX - d_rawImageParameter.m_xPixelBeginOffset;
                int writeY = globalY - d_rawImageParameter.m_yPixelBeginOffset;

                if (writeX >= 0 && writeX < d_rawImageParameter.m_recImgWidth &&
                    writeY >= 0 && writeY < d_rawImageParameter.m_recImgHeight)
                {
                    int outIdx = disp * d_rawImageParameter.m_recImgHeight * d_rawImageParameter.m_recImgWidth +
                                 writeY * d_rawImageParameter.m_recImgWidth + writeX;

                    d_costVolFiltered[outIdx] = filteredValue / divide * multiply;
                }
            }
        }
    }
}

void CostVolFilter::costVolWindowFilter(const DataParameter &dataParameter, cv::Mat *costVol) {
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();
    FilterPatameter filterPatameter = dataParameter.getFilterPatameter();



    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 启动 CUDA 核函数
    cudaEventRecord(start); // 记录开始时间
    dim3 blockDim(16, 16, 4);  // z方向处理 disparity，8x8x4=256 threads
    dim3 gridDim(rawImageParameter.m_xLensNum, rawImageParameter.m_yLensNum);
    
    costVolWindowFilterKernel<<<gridDim, blockDim>>>(
        d_microImageParameter,
        d_filterPatameterDevice,
        disparityParameter.m_disNum
    );
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