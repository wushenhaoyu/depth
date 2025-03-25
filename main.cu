#include "ToolTwoTestDemo.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono> // 包含 chrono 库
//#include <opencv2/core/cuda.hpp>
using namespace cv;
using namespace std;

#include <cuda_runtime.h>

void printDeviceProperties(int devID) {
    cudaDeviceProp deviceProps;
    cudaError_t err;

    // 获取设备属性
    err = cudaGetDeviceProperties(&deviceProps, devID);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 打印设备属性
    std::cout << "运行GPU设备: " << deviceProps.name << std::endl;
    std::cout << "SM数量: " << deviceProps.multiProcessorCount << std::endl;
    std::cout << "L2缓存大小: " << deviceProps.l2CacheSize / (1024 * 1024) << "M" << std::endl;
    std::cout << "SM最大驻留线程数量: " << deviceProps.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "设备是否支持流优先级: " << (deviceProps.streamPrioritiesSupported ? "是" : "否") << std::endl;
    std::cout << "设备是否支持在L1缓存中缓存全局内存: " << (deviceProps.globalL1CacheSupported ? "是" : "否") << std::endl;
    std::cout << "设备是否支持在L1缓存中缓存本地内存: " << (deviceProps.localL1CacheSupported ? "是" : "否") << std::endl;
    std::cout << "一个SM可用的最大共享内存量: " << deviceProps.sharedMemPerMultiprocessor / 1024 << "KB" << std::endl;
    std::cout << "一个SM可用的32位最大寄存器数量: " << deviceProps.regsPerMultiprocessor / 1024 << "K" << std::endl;
    std::cout << "一个SM最大驻留线程块数量: " << deviceProps.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "GPU内存带宽: " << deviceProps.memoryBusWidth << std::endl;
    std::cout << "GPU内存频率: " << (float)deviceProps.memoryClockRate / (1024 * 1024) << "GHz" << std::endl;
}



void demo2()
{
	ToolTwoTestDemo toolTwoTestDemo;
	toolTwoTestDemo.data5compute();
}


int main()
{
	bool hascuda = cv::cuda::getCudaEnabledDeviceCount() > 0;
	if(hascuda)
	{
		std::cout<<"Cuda available"<<std::endl;
	}
	else{
		std::cout<<"No cuda!"<<std::endl;
	}
	printDeviceProperties(0);
	// 记录开始时间
	auto start = std::chrono::high_resolution_clock::now();

	//demo1();
	demo2();
	// 读取灰度图像
	
	/*
	string path = "C:/work/LightFieldTool_CPU/LightFieldTool_CPU/205_66/66_5.bmp";
	cv::Mat m_inputImg = imread(path, IMREAD_COLOR);
	 
	//cv::Mat m_inputImg = cv::imread(path, cv::IMREAD_GRAYSCALE);
	if (m_inputImg.empty()) {
		std::cerr << "Error: Unable to open image." << std::endl;
		return -1;
	}
	m_inputImg.convertTo(m_inputImg, CV_32FC3, 1 / 255.0f);   //三通道彩色图像处理
	
	cv::Mat im_gray, tmp;
	// 输出图像，用于存储水平梯度
	cv::Mat m_gradImg,dst_x, dst_y, dst;
	m_inputImg.convertTo(tmp, CV_32F);
	cv::cvtColor(tmp, im_gray, COLOR_RGB2GRAY); //先移除这个，因为本来就是要处理灰度图像（然而并不行,因为后面颜色梯度方面需要用到）
	//cv::Sobel(m_inputImg, m_gradImg, CV_32F, 1, 0, 1);
	Sobel(im_gray, dst_x, CV_32F, 1, 0);
	Sobel(im_gray, dst_y, CV_32F, 0, 1);
	addWeighted(dst_x, 0.5, dst_y, 0.5, 0, dst);
	//convertScaleAbs(dst, dst);
	//cv::Sobel(im_gray, m_gradImg, CV_32F, 1, 0, 1);
	m_gradImg += 0.5;




	string outputpath = "C:/Users/ZHY/Desktop/m_gradImg_1m.png";


	cv::namedWindow("input", cv::WINDOW_NORMAL);


	// 显示结果
	//cv::imshow("input", m_gradImg);
	cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX);
	dst.convertTo(dst, CV_8U);
	cv::imshow("input", dst);
	imwrite(outputpath, dst);




	cv::waitKey(0);
	*/

	// 记录结束时间
	auto end = std::chrono::high_resolution_clock::now();

	// 计算时间差
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "程序运行时间: " << duration.count() << " 毫秒" << std::endl;
	cudaDeviceReset();
	return 0;

//注意，边缘检测因为只有左右没有上下，导致了对于有这明显上下分布的物体没法检测到边缘
}