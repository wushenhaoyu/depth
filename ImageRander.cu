#include "ImageRander.h"
#include "DataParameter.h"
#include <iomanip>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;
using namespace cv;

//修改了WIDTH和HEIGHT,此处参数只能影响到渲染图的生成效果
#define MEAN_DISP_LEN_RADIUS 18//平均距离长度 8 注意该参数在需要让计算深度的点尽量都在一个圆内(方型)  10
#define PATCH_SCALE9 9//路径比例 9
#define RANDER_SCALE 0.9//渲染比例 render  0.35
#define DEST_WIDTH 44//38 27 44
#define DEST_HEIGHT 44//38 27 44


ImageRander::ImageRander()
{

}

ImageRander::~ImageRander()
{

}

void ImageRander::imageRanderWithMask(const DataParameter &dataParameter, cv::Mat &rawDisp, cv::Mat *confidentMask)
{//对带置信度mask的情况进行子孔径渲染
	RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
	MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
	DisparityParameter disparityParameter = dataParameter.getDisparityParameter();

	cv::Mat tmpRawDisp = cv::Mat::zeros(rawDisp.rows, rawDisp.cols, CV_32FC1);
	rawDisp.convertTo(tmpRawDisp, CV_32F, disparityParameter.m_dispStep, disparityParameter.m_dispMin);

	float **ppLensMeanDisp = new float*[rawImageParameter.m_yLensNum];
	for (int i = 0; i < rawImageParameter.m_yLensNum; i++)
		ppLensMeanDisp[i] = new float[rawImageParameter.m_xLensNum];

//#pragma omp parallel for
	for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
	{
		for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
		{
			Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];
			int x_begin = curCenterPos.x - rawImageParameter.m_xPixelBeginOffset - MEAN_DISP_LEN_RADIUS;
			int y_begin = curCenterPos.y - rawImageParameter.m_yPixelBeginOffset - MEAN_DISP_LEN_RADIUS;

			cv::Mat srcCost = tmpRawDisp(cv::Rect(x_begin, y_begin, MEAN_DISP_LEN_RADIUS * 2 + 1, MEAN_DISP_LEN_RADIUS * 2 + 1));
			cv::Mat mask = (*confidentMask)(cv::Rect(x_begin, y_begin, MEAN_DISP_LEN_RADIUS * 2 + 1, MEAN_DISP_LEN_RADIUS * 2 + 1));
			ppLensMeanDisp[y][x] = std::max(cv::mean(srcCost, mask)[0], (double)(disparityParameter.m_dispMin));
		}
	}

	//渲染开始
	cv::Mat randerMapinput, randerSceneMap, finalRanderMap;
	//dataParameter.m_inputImgRec.convertTo(randerMapinput, CV_64FC3);
	dataParameter.m_inputImgRec.convertTo(randerMapinput, CV_64FC1);//转为双精度浮点数单通道
	imageRander(ppLensMeanDisp, rawImageParameter, microImageParameter, randerMapinput, randerSceneMap);
	std::string storeName = dataParameter.m_folderPath + "/randerSceneMap.bmp";
	//randerSceneMap.convertTo(finalRanderMap, CV_8UC3);
	randerSceneMap.convertTo(finalRanderMap, CV_8UC1);
	imwrite(storeName, finalRanderMap);

	//这部分用来渲染深度图像，测试速度阶段暂时注释掉
//	/*
	cv::Mat sceneDisp = cv::Mat::zeros(rawDisp.rows, rawDisp.cols, CV_64FC1);
	rawDisp.convertTo(sceneDisp, CV_64FC1, disparityParameter.m_dispStep, disparityParameter.m_dispMin);
	cv::Mat randerDispMap, randerSparseDispMap;
	imageRander(ppLensMeanDisp, rawImageParameter, microImageParameter, sceneDisp, randerDispMap);
	randerDispMap.copyTo(randerSparseDispMap);
	storeName = dataParameter.m_folderPath + "/randerDispMap.bmp";
	dispMapShowForColor(storeName, randerDispMap);

	cv::Mat rawMask, randerSceneMask;
	(*confidentMask).convertTo(rawMask, CV_64FC1);
	imageRander(ppLensMeanDisp, rawImageParameter, microImageParameter, rawMask, randerSceneMask);
	cv::Mat finalRanderMask = cv::Mat::zeros(randerSceneMask.rows, randerSceneMask.cols, CV_8UC1);
//#pragma omp parallel for
	for (int py = 0; py < randerSceneMask.rows; py++)
	{
		double *ySceneData = (double *)randerSceneMask.ptr<double>(py);
		uchar *yFinalData = (uchar *)finalRanderMask.ptr<uchar>(py);
		for (int px = 0; px < randerSceneMask.cols; px++)
		{
			if (ySceneData[px] >= 255.0)
				yFinalData[px] = 255;
		}
	}
	storeName = dataParameter.m_folderPath + "/randerSceneMask.bmp";
	imwrite(storeName, finalRanderMask);

	outputSparseSceneDepth(dataParameter.m_folderPath, randerSparseDispMap, finalRanderMask);

	double minVal, maxVal;
	minMaxLoc(randerSparseDispMap, &minVal, &maxVal);
//#pragma omp parallel for
	for (int py = 0; py < randerSparseDispMap.rows; py++)
	{
		double *ySparseData = (double *)randerSparseDispMap.ptr<double>(py);
		uchar *yConfMask = (uchar *)finalRanderMask.ptr<uchar>(py);
		for (int px = 0; px < randerSparseDispMap.cols; px++)
		{
			if (yConfMask[px] == 0)
				ySparseData[px] = minVal;
		}
	}
	storeName = dataParameter.m_folderPath + "/randerSparseDispMap.bmp";
	dispMapShowForColor(storeName, randerSparseDispMap);
//  */
	for (int i = 0; i < rawImageParameter.m_yLensNum; i++)
		delete[]ppLensMeanDisp[i];
	delete[]ppLensMeanDisp;
}




void ImageRander::imageRander(float **ppLensMeanDisp, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, cv::Mat &randerImg, cv::Mat &destImg)
{
    // 分配内存用于存储渲染块
    RanderMapPatch **ppRanderMapPatch = new RanderMapPatch *[rawImageParameter.m_yLensNum];
    for (int i = 0; i < rawImageParameter.m_yLensNum; i++)
        ppRanderMapPatch[i] = new RanderMapPatch[rawImageParameter.m_xLensNum];

    // 并行处理每个微透镜中心的渲染块
#pragma omp parallel for
    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
    {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
        {
            int blockSize = fabs(std::round(ppLensMeanDisp[y][x]));
            Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];
            int starty = curCenterPos.y - blockSize / 2 - rawImageParameter.m_yPixelBeginOffset;
            int startx = curCenterPos.x - blockSize / 2 - rawImageParameter.m_xPixelBeginOffset;
            cv::Mat srcImg = randerImg(cv::Rect(startx, starty, blockSize, blockSize));
            ppRanderMapPatch[y][x].sy = curCenterPos.y;
            ppRanderMapPatch[y][x].sx = curCenterPos.x;
            cv::Mat tmp;
            cv::resize(srcImg, tmp, cv::Size(DEST_WIDTH, DEST_HEIGHT), 0, 0, cv::INTER_LINEAR);
            cv::flip(tmp, ppRanderMapPatch[y][x].simg, -1);
        }
    }

    // 计算渲染图的范围
    int sx_begin = INT_MAX, sy_begin = INT_MAX;
    int sx_end = INT_MIN, sy_end = INT_MIN;

    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
    {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
        {
            sy_begin = std::min(sy_begin, ppRanderMapPatch[y][x].sy - ppRanderMapPatch[y][x].simg.rows / 2);
            sx_begin = std::min(sx_begin, ppRanderMapPatch[y][x].sx - ppRanderMapPatch[y][x].simg.cols / 2);
            sy_end = std::max(sy_end, ppRanderMapPatch[y][x].sy + ppRanderMapPatch[y][x].simg.rows / 2);
            sx_end = std::max(sx_end, ppRanderMapPatch[y][x].sx + ppRanderMapPatch[y][x].simg.cols / 2);
        }
    }

    int randerMapWidth = sx_end - sx_begin + 1;
    int randerMapHeight = sy_end - sy_begin + 1;

    // 创建渲染图和计数图
    cv::Mat randerMap = cv::Mat::zeros(randerMapHeight, randerMapWidth, randerImg.type());
    cv::Mat randerCount = cv::Mat::zeros(randerMapHeight, randerMapWidth, CV_64FC1);
    cv::Mat tmpCount = cv::Mat::ones(DEST_HEIGHT * 2, DEST_WIDTH * 2, CV_64FC1); // 修改此处的参数值

    // 将每个渲染块累加到渲染图中
    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
    {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
        {
            int sy_b = ppRanderMapPatch[y][x].sy - ppRanderMapPatch[y][x].simg.rows / 2 - sy_begin;
            int sy_e = ppRanderMapPatch[y][x].sy + ppRanderMapPatch[y][x].simg.rows / 2;
            int sx_b = ppRanderMapPatch[y][x].sx - ppRanderMapPatch[y][x].simg.cols / 2 - sx_begin;
            int sx_e = ppRanderMapPatch[y][x].sx + ppRanderMapPatch[y][x].simg.cols / 2;

            cv::Mat randerMapRect = randerMap(cv::Rect(sx_b, sy_b, ppRanderMapPatch[y][x].simg.cols, ppRanderMapPatch[y][x].simg.rows));
            cv::Mat randerCountRect = randerCount(cv::Rect(sx_b, sy_b, ppRanderMapPatch[y][x].simg.cols, ppRanderMapPatch[y][x].simg.rows));
            cv::Mat randerMapCountTmp = tmpCount(cv::Rect(0, 0, ppRanderMapPatch[y][x].simg.cols, ppRanderMapPatch[y][x].simg.rows));

            randerMapRect = randerMapRect + ppRanderMapPatch[y][x].simg;
            randerCountRect = randerCountRect + randerMapCountTmp;
        }
    }

    // 对渲染图进行归一化
    for (int y = 0; y < randerMapHeight; y++)
    {
        double *yRanderData = (double *)randerMap.ptr<double>(y);
        double *yRanderCount = (double *)randerCount.ptr<double>(y);
        for (int x = 0; x < randerMapWidth; x++)
        {
            if (yRanderCount[x] < 1.0)
                continue;
            else
            {
                if (randerMap.channels() == 3)
                {
                    yRanderData[3 * x] /= yRanderCount[x];
                    yRanderData[3 * x + 1] /= yRanderCount[x];
                    yRanderData[3 * x + 2] /= yRanderCount[x];
                }
                else
                    yRanderData[x] /= yRanderCount[x];
            }
        }
    }

    // 去除边界黑色空洞
    int left = rawImageParameter.m_xCenterBeginOffset;
    int right = rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset - 1;
    int top = rawImageParameter.m_yCenterBeginOffset;
    int below = rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset - 1;

    int x_left = 0;
    int x_right = randerMapWidth;
    int y_top = 0;
    int y_below = randerMapHeight;

    for (int y = top; y <= below; y++)
    {
        x_left = std::max(int(ppRanderMapPatch[y][left].sx - ppRanderMapPatch[y][left].simg.cols / 2 - sx_begin), x_left);
        x_right = std::min(int(ppRanderMapPatch[y][right].sx - ppRanderMapPatch[y][right].simg.cols / 2 - sx_begin + ppRanderMapPatch[y][right].simg.cols), x_right);
    }
    for (int x = left; x <= right; x++)
    {
        y_top = std::max(int(ppRanderMapPatch[top][x].sy - ppRanderMapPatch[top][x].simg.rows / 2 - sy_begin), y_top);
        y_below = std::min(int(ppRanderMapPatch[below][x].sy - ppRanderMapPatch[below][x].simg.rows / 2 - sy_begin + ppRanderMapPatch[below][x].simg.rows), y_below);
    }

    cv::Mat repairMap;
    randerMap(cv::Rect(x_left, y_top, x_right - x_left, y_below - y_top)).copyTo(repairMap);

    // 调整最终输出图像的大小
    cv::resize(repairMap, destImg, cv::Size(0, 0), RANDER_SCALE, RANDER_SCALE, cv::INTER_CUBIC);

    // 释放内存
    for (int i = 0; i < rawImageParameter.m_yLensNum; i++)
        delete[] ppRanderMapPatch[i];
    delete[] ppRanderMapPatch;
}


void ImageRander::outputSparseSceneDepth(string folderName, cv::Mat &sceneSparseDepth, cv::Mat &sceneDepthMask)
{
	std::string storeName;
	ofstream ofs1;
	storeName = folderName + "/sceneInitDisp.txt";
	ofs1.open(storeName, ofstream::out);
	ofstream ofs2;
	storeName = folderName + "/sceneInitDispMask.txt";
	ofs2.open(storeName, ofstream::out);

	for (int y = 0; y < sceneSparseDepth.rows; ++y)
	{
		double *pYSceneDisp = (double *)sceneSparseDepth.ptr<double>(y);
		uchar *pYMask = (uchar *)sceneDepthMask.ptr<uchar>(y);
		for (int x = 0; x < sceneSparseDepth.cols; ++x)
		{
			ofs1 << fixed << setprecision(5) << pYSceneDisp[x] << " ";
			if (pYMask[x] == 255)
				ofs2 << 1 << " ";
			else
				ofs2 << 0 << " ";
			//ofs2 << int(pYMask[x]) << " ";
		}
		ofs1 << endl;
		ofs2 << endl;
	}
	ofs1.close();
	ofs2.close();
}


void ImageRander::imageRanderWithOutMask(const DataParameter &dataParameter, cv::Mat &rawDisp)
{//对没有置信度mask的情况进行子孔径渲染
	RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
	MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
	DisparityParameter disparityParameter = dataParameter.getDisparityParameter();

	cv::Mat tmpRawDisp = cv::Mat::zeros(rawDisp.rows, rawDisp.cols, CV_32FC1);
	rawDisp.convertTo(tmpRawDisp, CV_32F, disparityParameter.m_dispStep, disparityParameter.m_dispMin);

	float **ppLensMeanDisp = new float*[rawImageParameter.m_yLensNum];
	for (int i = 0; i < rawImageParameter.m_yLensNum; i++)
		ppLensMeanDisp[i] = new float[rawImageParameter.m_xLensNum];

#pragma omp parallel for
	for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
	{
		for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
		{
			Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];
			int x_begin = curCenterPos.x - rawImageParameter.m_xPixelBeginOffset - MEAN_DISP_LEN_RADIUS;
			int y_begin = curCenterPos.y - rawImageParameter.m_yPixelBeginOffset - MEAN_DISP_LEN_RADIUS;
			cv::Mat srcCost = tmpRawDisp(cv::Rect(x_begin, y_begin, MEAN_DISP_LEN_RADIUS * 2 + 1, MEAN_DISP_LEN_RADIUS * 2 + 1));
			ppLensMeanDisp[y][x] = std::max(cv::mean(srcCost)[0], (double)(disparityParameter.m_dispMin));
		}
	}

	//渲染开始
	cv::Mat randerMapinput, randerSceneMap, finalRanderMap;
	dataParameter.m_inputImgRec.convertTo(randerMapinput, CV_64FC3);
	imageRander(ppLensMeanDisp, rawImageParameter, microImageParameter, randerMapinput, randerSceneMap);
	std::string storeName = dataParameter.m_folderPath + "/randerSceneMap.bmp";
	randerSceneMap.convertTo(finalRanderMap, CV_8UC3);
	imwrite(storeName, finalRanderMap);


//	/*
	cv::Mat sceneDisp = cv::Mat::zeros(rawDisp.rows, rawDisp.cols, CV_64FC1);
	rawDisp.convertTo(sceneDisp, CV_64FC1, disparityParameter.m_dispStep, disparityParameter.m_dispMin);
	cv::Mat randerDispMap, randerSparseDispMap;
	imageRander(ppLensMeanDisp, rawImageParameter, microImageParameter, sceneDisp, randerDispMap);
	randerDispMap.copyTo(randerSparseDispMap);
	storeName = dataParameter.m_folderPath + "/randerDispMap.bmp";
	dispMapShowForColor(storeName, randerDispMap);

//	*/
}