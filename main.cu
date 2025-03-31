#include "ToolOneTestDemo.h"
#include "ToolTwoTestDemo.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono> // 包含 chrono 库
//#include <opencv2/core/cuda.hpp>
using namespace cv;
using namespace std;





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
	// 记录开始时间
	 
		auto start = std::chrono::high_resolution_clock::now();
		demo2();
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "程序运行时间: " << duration.count() << " 毫秒" << std::endl;
		
	

	//demo1();
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


	// 计算时间差
	return 0;

//注意，边缘检测因为只有左右没有上下，导致了对于有这明显上下分布的物体没法检测到边缘
}