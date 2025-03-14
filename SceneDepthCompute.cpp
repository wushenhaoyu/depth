#include "SceneDepthCompute.h"
#include "DataParameter.h"
#include <iomanip>

using namespace cv;
using namespace std;

SceneDepthCompute::SceneDepthCompute()
{

}

SceneDepthCompute::~SceneDepthCompute()
{

}

void SceneDepthCompute::loadSceneDataCost(const DataParameter &dataParameter, std::string referImgName, std::string referDispXmlName,
	std::string referMaskXmlName, std::string mappingFileName)
{//新的对应视差的载入
	RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
	MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
	DisparityParameter disparityParameter = dataParameter.getDisparityParameter();

	cv::Mat subAperatureImg = imread(dataParameter.m_folderPath + '/' + referImgName, IMREAD_COLOR);
	int img_width = subAperatureImg.cols;
	int img_height = subAperatureImg.rows;

	Mat referDispMap = Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_8UC1);
	readDispMapFromXML(dataParameter.m_folderPath + '/' + referDispXmlName, referDispMap);
	Mat referDispMask = Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_8UC1);
	readDispMapFromXML(dataParameter.m_folderPath + '/' + referMaskXmlName, referDispMask);

	Mat sceneDispMap = Mat::zeros(img_height, img_width, CV_32FC1);
	Mat sceneDispMask = Mat::zeros(img_height, img_width, CV_8UC1);

	ifstream dispCostIn(dataParameter.m_folderPath + '/' + mappingFileName);
	int coor_value_num = 0;
	dispCostIn >> coor_value_num;

	for (int y = 0; y < img_height; ++y)
	{
		float *pYSceneDisp = (float *)sceneDispMap.ptr<float>(y);
		uchar *pYMask = (uchar *)sceneDispMask.ptr<uchar>(y);

		for (int x = 0; x < img_width; ++x)
		{
			float tmpSumCost = 0.0, tmpNumCount = 0.0;
			float validNumCount = 0.0;
			for (int z = 0; z < coor_value_num; ++z)
			{
				int ty, tx;
				dispCostIn >> ty >> tx;
				if (ty > 0 && ty - rawImageParameter.m_yPixelBeginOffset >= 0 && ty - rawImageParameter.m_yPixelBeginOffset < rawImageParameter.m_recImgHeight 
					&& tx - rawImageParameter.m_xPixelBeginOffset >= 0 && tx - rawImageParameter.m_xPixelBeginOffset < rawImageParameter.m_recImgWidth){
					if (referDispMask.at<uchar>(ty - rawImageParameter.m_yPixelBeginOffset, tx - rawImageParameter.m_xPixelBeginOffset) == 255){
						tmpSumCost += (referDispMap.at<uchar>(ty - rawImageParameter.m_yPixelBeginOffset, tx - rawImageParameter.m_xPixelBeginOffset)*
							disparityParameter.m_dispStep + disparityParameter.m_dispMin);
						tmpNumCount += 1.0f;
					}
					validNumCount += 1.0f;
				}
			}
			if (tmpNumCount >= 1.0 && tmpNumCount > validNumCount / 2.0){
				pYSceneDisp[x] = tmpSumCost / tmpNumCount;
				pYMask[x] = 1;
			}
			else
				pYSceneDisp[x] = 0;
		}
	}

	std::string storeName = dataParameter.m_folderPath + "/initGraySceneDepth.bmp";
	dispMapShow(storeName, sceneDispMap);
	storeName = dataParameter.m_folderPath + "/initColorSceneDepth.bmp";
	dispMapShowForColor(storeName, sceneDispMap);

	storeName = dataParameter.m_folderPath + "/initSceneMask.bmp";
	dispMapShow(storeName, sceneDispMask);

	ofstream ofs1;
	storeName = dataParameter.m_folderPath + "/initSceneDisp.txt";
	ofs1.open(storeName, ofstream::out);
	ofstream ofs2;
	storeName = dataParameter.m_folderPath + "/initSceneDispMask.txt";
	ofs2.open(storeName, ofstream::out);

	for (int y = 0; y < img_height; ++y)
	{
		float *pYSceneDisp = (float *)sceneDispMap.ptr<float>(y);
		uchar *pYMask = (uchar *)sceneDispMask.ptr<uchar>(y);
		for (int x = 0; x < img_width; ++x)
		{
			ofs1 << fixed << setprecision(5) << pYSceneDisp[x] << " ";
			ofs2 << int(pYMask[x]) << " ";
		}
		ofs1 << endl;
		ofs2 << endl;
	}
	dispCostIn.close();
	ofs1.close();
	ofs2.close();
}

void SceneDepthCompute::outputMicrolensDisp(const DataParameter &dataParameter, Mat &rawDisp, Mat *confidentMask)
{//输出微透镜图像中平均视差文件
	RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
	MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
	DisparityParameter disparityParameter = dataParameter.getDisparityParameter();

	string storeName;
	ofstream ofs1;
	storeName = dataParameter.m_folderPath + "/white_center_crop.txt";
	ofs1.open(storeName, ofstream::out);
	ofstream ofs2;
	storeName = dataParameter.m_folderPath + "/center_disparity_crop.txt";
	ofs2.open(storeName, ofstream::out);

	for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
	{
		for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
		{
			Point2d &centerPos = microImageParameter.m_ppLensCenterPoints[y][x];
			float tempSmoothValue = getSmoothValue(rawImageParameter, disparityParameter, rawDisp, centerPos, confidentMask, 9);
			ofs1 << centerPos.y << " " << centerPos.x << " ";
			ofs2 << tempSmoothValue << " ";
		}
		ofs1 << endl;
		ofs2 << endl;
	}
	ofs1.close();
	ofs2.close();
}

float SceneDepthCompute::getSmoothValue(const RawImageParameter &rawImageParameter, const DisparityParameter &disparityParameter, 
	Mat &rawDisp, Point2d &centerPos, Mat *confidentMask, const int shift_size)
{
	float tempSum = 0.0, tempCount = 0.0;
	float uncertaintyValue = 0.0;
	for (int shift_y = -shift_size; shift_y <= shift_size; shift_y++)
	{
		for (int shift_x = -shift_size; shift_x <= shift_size; shift_x++)
		{
			int ty = centerPos.y + shift_y - rawImageParameter.m_yPixelBeginOffset;
			int tx = centerPos.x + shift_x - rawImageParameter.m_xPixelBeginOffset;

			if (confidentMask != nullptr)
			{
				if (confidentMask->at<uchar>(ty, tx) == 255){
					tempSum += disparityParameter.m_dispStep*static_cast<float>(rawDisp.at<uchar>(ty, tx)) + disparityParameter.m_dispMin;
					tempCount += 1.0;
				}
			}
			else
			{
				tempSum += disparityParameter.m_dispStep*static_cast<float>(rawDisp.at<uchar>(ty, tx)) + disparityParameter.m_dispMin;
				tempCount += 1.0;
			}

			uncertaintyValue += disparityParameter.m_dispStep*static_cast<float>(rawDisp.at<uchar>(ty, tx)) + disparityParameter.m_dispMin;
		}
	}

	if (tempCount < 1.0f)
		return uncertaintyValue / ((2 * shift_size + 1)*(2 * shift_size + 1));
	else
		return tempSum / tempCount;
}

void SceneDepthCompute::loadSceneDataCost(const DataParameter &dataParameter, cv::Mat &subApertureImg, std::string mappingFileName,
	cv::Mat *srcCostVol, cv::Mat *destCostVol)
{//新的对应视差的载入
	RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
	DisparityParameter disparityParameter = dataParameter.getDisparityParameter();

	int img_width = subApertureImg.cols;
	int img_height = subApertureImg.rows;
	Point2i tempIndex;
	ifstream datain(dataParameter.m_folderPath + "/" + mappingFileName);
	int coor_value_num = 0;
	datain >> coor_value_num;
	for (int y = 0; y < img_height; y++)
	{
		for (int x = 0; x < img_width; x++)
		{
			vector<Point2i> tempVecIndex;
			for (int z = 0; z < coor_value_num; z++)
			{
				datain >> tempIndex.y >> tempIndex.x;
				if (tempIndex.y > 0)
					tempVecIndex.push_back(tempIndex);
			}
			for (int d = 0; d < disparityParameter.m_disNum; d++)
			{
				float* cost = (float*)destCostVol[d].ptr<float>(y);
				float tmpValue = 0.0;
				for (int z = 0; z < tempVecIndex.size(); z++){
					float* tempCost = (float*)srcCostVol[d].ptr<float>(tempVecIndex[z].y - rawImageParameter.m_yPixelBeginOffset);
					tmpValue += tempCost[tempVecIndex[z].x - rawImageParameter.m_xPixelBeginOffset];
				}
				if (tempVecIndex.size() == 0)
					cost[x] = -1;
				else
					cost[x] = tmpValue / tempVecIndex.size();
			}
		}//for x
	}//for y
	datain.close();
	fillOtherCostVol(img_height, img_width, disparityParameter.m_disNum, destCostVol);
}

void SceneDepthCompute::fillOtherCostVol(int height, int width, int maxDis, Mat* &costVol)
{//对一些没有cost的值就行填补
	double minVal, maxVal = 0.0, tempMaxVal;
	for (int i = 0; i < maxDis; i++)
	{
		minMaxLoc(costVol[i], &minVal, &tempMaxVal);
		if (tempMaxVal > maxVal)
			maxVal = tempMaxVal;
	}

	int tempConutFill = 0;
	for (int d = 0; d < maxDis; d++)
	{
		for (int y = 0; y < height; y++)
		{
			float* cost = (float*)costVol[d].ptr<float>(y);
			for (int x = 0; x < width; x++)
			{
				if (cost[x] < 0)
				{
					if ((y>0 && y < height - 1) && (x>0 && x < width - 1))
					{
						float* costy1 = (float*)costVol[d].ptr<float>(y - 1);
						float* costy2 = (float*)costVol[d].ptr<float>(y + 1);

						float y1 = costy1[x] < 0 ? maxVal : costy1[x];
						float y2 = costy2[x] < 0 ? maxVal : costy2[x];
						float x1 = cost[x - 1] < 0 ? maxVal : cost[x - 1];
						float x2 = cost[x + 1] < 0 ? maxVal : cost[x + 1];
						cost[x] = (y1 + y2 + x1 + x2) / 4;
					}
					else if ((y <= 0 || y >= height - 1) && (x > 0 && x < width - 1))
					{
						float x1 = cost[x - 1] < 0 ? maxVal : cost[x - 1];
						float x2 = cost[x + 1] < 0 ? maxVal : cost[x + 1];
						cost[x] = (x1 + x2) / 2;
					}
					else if ((y>0 && y < height - 1) && (x <= 0 && x >= width - 1))
					{
						float* costy1 = (float*)costVol[d].ptr<float>(y - 1);
						float* costy2 = (float*)costVol[d].ptr<float>(y + 1);

						float y1 = costy1[x] < 0 ? maxVal : costy1[x];
						float y2 = costy2[x] < 0 ? maxVal : costy2[x];

						cost[x] = (y1 + y2) / 2;
					}
					else
					{
						cost[x] = maxVal;
					}
					tempConutFill++;
				}
			}
		}
	}
}