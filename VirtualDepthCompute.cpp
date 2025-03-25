#include "VirtualDepthCompute.h"
#include "DataParameter.cuh"
#include <algorithm>

#define BASE_LINE_DIS 34.0

using namespace cv;
using namespace std;

VirtualDepthCompute::VirtualDepthCompute()
{
}

VirtualDepthCompute::~VirtualDepthCompute()
{
}

void VirtualDepthCompute::virtualDepthCompute(const DataParameter &dataParameter, cv::Mat &referDispMap, cv::Mat &referDiskMask)
{//�����������
	RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
	MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
	DisparityParameter disparityParameter = dataParameter.getDisparityParameter();

	for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
	{
		for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
		{
			Point2d &centerPos = microImageParameter.m_ppLensCenterPoints[y][x];
			int curCenterIndex = y*rawImageParameter.m_xLensNum + x;
			for (int py = centerPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
				py <= centerPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; py++)
			{
				for (int px = centerPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
					px <= centerPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; px++)
				{
					if (microImageParameter.m_ppPixelsMappingSet[py][px] == curCenterIndex){
						uchar *yDispMat = (uchar *)referDispMap.ptr<uchar>(py - rawImageParameter.m_yPixelBeginOffset);
						uchar *yDispMask = (uchar *)referDiskMask.ptr<uchar>(py - rawImageParameter.m_yPixelBeginOffset);

						if (yDispMask[px - rawImageParameter.m_xPixelBeginOffset] == 255){
							int curDispIndex = yDispMat[px - rawImageParameter.m_xPixelBeginOffset];
							float curDisp = disparityParameter.m_dispStep*curDispIndex + disparityParameter.m_dispMin;//��Ӧʵ���Ӳ�
							float curVirtualDepth = BASE_LINE_DIS / curDisp;//���߳��ȳ�ʵ���Ӳ�
							//����������ȵ������x,y����
							float curVirtualX = centerPos.x + (centerPos.x - px)*curVirtualDepth;//Xv = Cx+(Cx-Xr)*Vz
							float curVirtualY = centerPos.y + (centerPos.y - py)*curVirtualDepth;//Yv = Cy+(Cy-Yr)*Vz
							m_pVirtualPointVec.push_back(Point3f(curVirtualX, curVirtualY, curVirtualDepth));
						}
					}
				}
			}
		}
	}

	//virtualDepthPointStore(dataParameter.m_folderPath, "virtualDepthPoints.xml");
	virtualDepthImageCreat(dataParameter.m_folderPath, "virtualDepthImage.bmp");
}

/*
void VirtualDepthCompute::virtualDepthPointStore(std::string pathName, std::string storeFileName)
{//����ά�������
	std::string storeName = pathName + "/VirtualDepth/" + storeFileName;
	FileStorage fs(storeName, FileStorage::WRITE); //����XML�ļ�  
	if (!fs.isOpened())
	{
		cerr << "failed to open " << storeName << endl;
	}
	fs << "VECTOR" << "["; // ע��Ҫ��������  
	for (vector<Point3f>::iterator it = m_pVirtualPointVec.begin(); it != m_pVirtualPointVec.end(); it++)
	{
		fs << (*it);
	}
	fs << "]";
	fs.release();
}
*/

void VirtualDepthCompute::virtualDepthImageCreat(std::string pathName, std::string storeImgName)
{//�����������ͼ��
	float vx_begin = m_pVirtualPointVec[0].x, vy_begin = m_pVirtualPointVec[0].y, vx_end = m_pVirtualPointVec[0].x, vy_end = m_pVirtualPointVec[0].y;

	for (int i = 0; i < m_pVirtualPointVec.size(); i++)
	{
		vx_begin = std::min(vx_begin, m_pVirtualPointVec[i].x);
		vx_end = std::max(vx_end, m_pVirtualPointVec[i].x);
		vy_begin = std::min(vy_begin, m_pVirtualPointVec[i].y);
		vy_end = std::max(vy_end, m_pVirtualPointVec[i].y);
	}

	int image_xOffset = round(vx_begin) - 0;
	int image_yOffset = round(vy_begin) - 0;
	int image_width = round(vx_end) - round(vx_begin) + 1;
	int image_height = round(vy_end) - round(vy_begin) + 1;

	Mat virtualMat = Mat::zeros(image_height, image_width, CV_32FC1);

	int **countArray = new int *[image_height];
	for (int j = 0; j < image_height; j++)
	{
		countArray[j] = new int[image_width];
		memset(countArray[j], 0, image_width*sizeof(int));
	}

	for (int i = 0; i < m_pVirtualPointVec.size(); i++)
	{
		int y = round(m_pVirtualPointVec[i].y) - image_yOffset;
		int x = round(m_pVirtualPointVec[i].x) - image_xOffset;
		float *yRows = (float *)virtualMat.ptr<float>(y);
		//yRows[x] += m_pVirtualPointVec[i].vz;
		yRows[x] = std::max(float(1.0) / m_pVirtualPointVec[i].z, yRows[x]);
		countArray[y][x]++;
	}

	// 	for (int y = 0; y < image_height; y++)
	// 	{
	// 		float *yRows = (float *)virtualMat.ptr<float>(y);
	// 		for (int x = 0; x < image_width; x++)
	// 		{
	// 			if (countArray[y][x] != 0)
	// 			{
	// 				yRows[x] /= countArray[y][x];
	// 				yRows[x] = 1.0 / yRows[x];
	// 			}
	// 		}
	// 	}
	float tempDepthMin = 100000.0, tempDepthMax = 0.0;
	for (int y = 0; y < image_height; y++)
	{
		float *yRows = (float *)virtualMat.ptr<float>(y);
		for (int x = 0; x < image_width; x++)
		{
			if (countArray[y][x] != 0)
			{
				if (yRows[x] < tempDepthMin) tempDepthMin = yRows[x];
				if (yRows[x] > tempDepthMax) tempDepthMax = yRows[x];
			}
		}
	}

	float tempDepthMiddle = (tempDepthMin + tempDepthMax) / 2.0;
//#pragma omp parallel for
	for (int y = 0; y < image_height; y++)
	{
		float *yRows = (float *)virtualMat.ptr<float>(y);
		for (int x = 0; x < image_width; x++)
		{
			if (countArray[y][x] == 0)
			{
				yRows[x] = tempDepthMiddle;
			}
		}
	}

	double minVal; double maxVal;
	minMaxLoc(virtualMat, &minVal, &maxVal);
	Mat disp;
	virtualMat.convertTo(disp, CV_8UC1, 255.0 / (maxVal - minVal), -minVal*255.0 / (maxVal - minVal));
	cv::Mat falseColorsMap;
	applyColorMap(disp, falseColorsMap, cv::COLORMAP_JET);//�Ӳ�Ҷ�ͼת���ɲ�ͼ,dispΪCV_8UC1����
	cout << "falseColorsMap.type = " << falseColorsMap.type() << endl;
	Mat m_pVirtualDepthImage = Mat::zeros(image_height, image_width, CV_8UC3);
//#pragma omp parallel for
	for (int y = 0; y < image_height; y++)
	{
		uchar *yDest = (uchar *)m_pVirtualDepthImage.ptr<uchar>(y);
		uchar *ySrc = (uchar *)falseColorsMap.ptr<uchar>(y);
		for (int x = 0; x < image_width; x++)
		{
			if (countArray[y][x] != 0)
			{
				yDest[3 * x + 0] = ySrc[3 * x + 0]; yDest[3 * x + 1] = ySrc[3 * x + 1]; yDest[3 * x + 2] = ySrc[3 * x + 2];
			}
		}
	}
	string storeName = pathName + "/VirtualDepth/" + storeImgName;
	imwrite(storeName, m_pVirtualDepthImage);
	for (int j = 0; j < image_height; j++)
		delete[]countArray[j];
	delete[]countArray;
}