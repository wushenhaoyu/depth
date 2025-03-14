#include "ConfidenceCompute.h"
#include "DataParameter.h"

using namespace std;
using namespace cv;

#define GRADIDENT_THRESHOLD  70//梯度区域阈值  //45~70:70
#define CIRCLE_GRAD_POINT_NUM 0 //小圆内有效梯度点的数目
#define MNN_CONFIDENT_MEASURE_THRES 10.0 //MNN指标置信度阈值   10
#define FINAL_CONFIDENT_MEASURE_THRES 35.0 //置信度二值化分割时的阈值 35
#define PI 3.1415926535898f

//#define RADIUS 15.5
#define RADIUS 8

ConfidenceCompute::ConfidenceCompute()
	:m_pGradientCircleMask(nullptr), m_pConfidentMask(nullptr)
{

}

ConfidenceCompute::~ConfidenceCompute()
{
	if (m_pGradientCircleMask != nullptr)
		delete m_pGradientCircleMask;
	if (m_pConfidentMask != nullptr)
		delete m_pConfidentMask;
}

void ConfidenceCompute::confidenceMeasureCompute(const DataParameter &dataParameter, cv::Mat *&costVol)
{//置信度计算开始
	RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
	MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
	DisparityParameter disparityParameter = dataParameter.getDisparityParameter();
	FilterPatameter filterPatameter = dataParameter.getFilterPatameter();

	m_folderPath = dataParameter.m_folderPath;
	cv::Mat inputRecImg;
	dataParameter.m_inputImgRec.convertTo(inputRecImg, CV_32FC3);
	gradientMeasureCompute(rawImageParameter, microImageParameter, filterPatameter, inputRecImg);
	confidentMeasureMMN(costVol, rawImageParameter, microImageParameter, disparityParameter, inputRecImg);
}

void ConfidenceCompute::gradientMeasureCompute(const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, 
	const FilterPatameter &filterPatameter, cv::Mat &srcImg)
{//梯度指标计算
	cv::Mat im_gray, dst_x, dst_y, dst, outPutImg;
	cvtColor(srcImg, im_gray, COLOR_RGB2GRAY);
	Sobel(im_gray, dst_x, CV_32F, 1, 0);
	Sobel(im_gray, dst_y, CV_32F, 0, 1);
	addWeighted(dst_x, 0.5, dst_y, 0.5, 0, dst);
	convertScaleAbs(dst, dst);
	cv::Mat src_grad = dst.clone();
	cv::Mat dst_grad = cv::Mat::zeros(src_grad.rows, src_grad.cols, CV_32FC1);
	src_grad.convertTo(src_grad, CV_32FC1);
	src_grad = src_grad.mul((*filterPatameter.m_pValidPixelsMask)(cv::Rect(rawImageParameter.m_xPixelBeginOffset, rawImageParameter.m_yPixelBeginOffset, 
		rawImageParameter.m_recImgWidth, rawImageParameter.m_recImgHeight)));

//#pragma omp parallel for
	for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
	{
		for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
		{
			//std::cout << "grad filter --- y=" << y << "\tx=" << x << std::endl;
			Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];
			int x_begin = curCenterPos.x - rawImageParameter.m_xPixelBeginOffset -  microImageParameter.m_circleDiameter/2 + microImageParameter.m_circleNarrow;//之前统一减9
			int y_begin = curCenterPos.y - rawImageParameter.m_yPixelBeginOffset - microImageParameter.m_circleDiameter /2 + microImageParameter.m_circleNarrow;
			int x_end = curCenterPos.x - rawImageParameter.m_xPixelBeginOffset + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow;
			int y_end = curCenterPos.y - rawImageParameter.m_yPixelBeginOffset + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow;
			cv::Mat divideMask = (*filterPatameter.m_pValidNeighborPixelsNum)(cv::Rect(x_begin + rawImageParameter.m_xPixelBeginOffset, 
				y_begin + rawImageParameter.m_yPixelBeginOffset, x_end - x_begin + 1, y_end - y_begin + 1));
			cv::Mat srcCost = src_grad(cv::Rect(x_begin, y_begin, x_end - x_begin + 1, y_end - y_begin + 1));
			cv::Mat destCost = dst_grad(cv::Rect(x_begin, y_begin, x_end - x_begin + 1, y_end - y_begin + 1));

			cv::filter2D(srcCost, destCost, -1, filterPatameter.m_filterKnernel, cv::Point(2, 2), 0, BORDER_CONSTANT);//cv:Point(-1,-1)改为2
			cv::divide(destCost, divideMask, destCost);
		}
	}
	dst_grad = dst_grad.mul((*filterPatameter.m_pValidPixelsMask)(cv::Rect(rawImageParameter.m_xPixelBeginOffset, rawImageParameter.m_yPixelBeginOffset,
		rawImageParameter.m_recImgWidth, rawImageParameter.m_recImgHeight)));
	double minVal, maxVal;
	minMaxLoc(dst_grad, &minVal, &maxVal);
	Mat dst_grad2;
	dst_grad.convertTo(dst_grad2, CV_8UC1, 255.0 / (maxVal - minVal), -minVal*255.0 / (maxVal - minVal));
	cv::threshold(dst_grad2, outPutImg, GRADIDENT_THRESHOLD, 255, THRESH_BINARY);//45~70

	m_pGradientCircleMask = new cv::Mat;
	*m_pGradientCircleMask = cv::Mat::zeros(rawImageParameter.m_yLensNum, rawImageParameter.m_xLensNum, CV_8UC1);
	gradientCircleSign(rawImageParameter, microImageParameter, outPutImg);

	cv::imwrite(m_folderPath + "/gradientMeasureMask.png", outPutImg);
}

void ConfidenceCompute::gradientCircleSign(const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, 
	cv::Mat &srcImg)
{//梯度圆mask标记
//#pragma omp parallel for
	for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
	{
		uchar *yCircleMask = (uchar*)(*m_pGradientCircleMask).ptr<uchar>(y);
		for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
		{
			//std::cout << "grad mask --- y=" << y << "\tx=" << x << std::endl;
			Point2d &centerPos = microImageParameter.m_ppLensCenterPoints[y][x];
			int curCenterIndex = y*rawImageParameter.m_xLensNum + x, sumCount = 0;
			for (int py = centerPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
				py <= centerPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; py++)
			{
				uchar *pYMask = (uchar *)srcImg.ptr<uchar>(py - rawImageParameter.m_yPixelBeginOffset);
				for (int px = centerPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
					px <= centerPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; px++)
				{
					if (microImageParameter.m_ppPixelsMappingSet[py][px] == curCenterIndex){
						if (pYMask[px - rawImageParameter.m_xPixelBeginOffset] > 20)
							++sumCount;
					}
				}
			}

			if (sumCount > CIRCLE_GRAD_POINT_NUM)
				yCircleMask[x] = 255;
		}
	}
}

void ConfidenceCompute::confidentMeasureMMN(cv::Mat *&costVol, const RawImageParameter &rawImageParameter, 
	const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter, cv::Mat &srcImg)
{//根据MMN的置信度指标计算
	cv::Mat confidentMat = cv::Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_32FC1);
	cv::Mat confidentMat2 = cv::Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_32FC1);
//#pragma omp parallel for
	for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
	{
		for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
		{
			//std::cout << "confident measure --- y=" << y << "\tx=" << x << std::endl;
			confidentMeasureMMN(costVol, rawImageParameter, microImageParameter, disparityParameter, confidentMat, y, x, confidentMat2);
		}
	}
	cv::Mat conftmp, conftmp2;
	confidentMat.convertTo(conftmp, CV_8UC1);
	string storeName = m_folderPath + "/conf_measure.png";
	cv::imwrite(storeName, conftmp);

	double minVal, maxVal;
	minMaxLoc(confidentMat2, &minVal, &maxVal);
	confidentMat2.convertTo(conftmp2, CV_8UC1, 255.0 / (maxVal - minVal), -minVal*255.0 / (maxVal - minVal));
	storeName = m_folderPath + "/final_conf_measure.png";
	cv::imwrite(storeName, conftmp2);

	storeName = m_folderPath + "/ConfidentMask.png";
	setConfidentMask(conftmp2, storeName, rawImageParameter, microImageParameter);
	
	Mat mask = Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_8UC1);
	confidentCircleJudge(conftmp2, mask, rawImageParameter, microImageParameter);
	drawConfidentCircle(costVol, mask, rawImageParameter, microImageParameter, disparityParameter, srcImg);

	std::string confidentImgStoreName = m_folderPath + "/confident_circle.png";
	lowTextureAreaPlot(rawImageParameter, microImageParameter, conftmp2, mask, confidentImgStoreName, CircleDrawMode::e_gray, true);
}

void ConfidenceCompute::confidentMeasureMMN(cv::Mat *&costVol, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, 
	const DisparityParameter &disparityParameter,cv::Mat &confMeasureMat, int y, int x, cv::Mat &confidentMat2)
{
	Point2d &centerPos = microImageParameter.m_ppLensCenterPoints[y][x];
	int curCenterIndex = y*rawImageParameter.m_xLensNum + x;

	float picConfMin = FLT_MAX, picConfMax = FLT_MIN;
	for (int py = centerPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
		py <= centerPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; py++)
	{
		for (int px = centerPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
			px <= centerPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; px++)
		{
			float *yConf = confMeasureMat.ptr<float>(py - rawImageParameter.m_yPixelBeginOffset);
			float *yConf2 = confidentMat2.ptr<float>(py - rawImageParameter.m_yPixelBeginOffset);
			float dCostMin = FLT_MAX;
			float dCostSec = FLT_MAX;
			float sumCost = 0.0;
			if (microImageParameter.m_ppPixelsMappingSet[py][px] == curCenterIndex){
				for (int d = 0; d < disparityParameter.m_disNum; d++)
				{
					float *cost = (float*)costVol[d].ptr<float>(py - rawImageParameter.m_yPixelBeginOffset);
					if (cost[px - rawImageParameter.m_xPixelBeginOffset] < dCostMin){
						dCostSec = dCostMin;
						dCostMin = cost[px - rawImageParameter.m_xPixelBeginOffset];
					}
					else if (cost[px - rawImageParameter.m_xPixelBeginOffset] > dCostMin && cost[px - rawImageParameter.m_xPixelBeginOffset] < dCostSec)
						dCostSec = cost[px - rawImageParameter.m_xPixelBeginOffset];

					sumCost += cost[px - rawImageParameter.m_xPixelBeginOffset];
				}
			}
			float confMMN = (dCostSec - dCostMin) / sumCost;//MMN置信度
			if (confMMN > picConfMax) picConfMax = confMMN;
			if (confMMN < picConfMin) picConfMin = confMMN;

			yConf[px - rawImageParameter.m_xPixelBeginOffset] = confMMN;
			yConf2[px - rawImageParameter.m_xPixelBeginOffset] = confMMN;
		}
	}

	for (int py = centerPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
		py <= centerPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; py++)
	{
		for (int px = centerPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
			px <= centerPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; px++)
		{
			float *yConf = confMeasureMat.ptr<float>(py - rawImageParameter.m_yPixelBeginOffset);
			if (microImageParameter.m_ppPixelsMappingSet[py][px] == curCenterIndex){
				yConf[px - rawImageParameter.m_xPixelBeginOffset] = (yConf[px - rawImageParameter.m_xPixelBeginOffset] - picConfMin) / (picConfMax - picConfMin) * 255.0;
			}
		}
	}
}

void ConfidenceCompute::setConfidentMask(cv::Mat &confidentMat, std::string confidentMaskName, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter)
{//设置置信度的mask
	m_pConfidentMask = new cv::Mat(confidentMat.rows, confidentMat.cols, CV_8UC1);

	cv::threshold(confidentMat, *m_pConfidentMask, FINAL_CONFIDENT_MEASURE_THRES, 255, THRESH_BINARY);
	confidentMaskRepair(*m_pConfidentMask, rawImageParameter, microImageParameter);
	cv::imwrite(confidentMaskName, *m_pConfidentMask);

	std::string storeName = m_folderPath + "/confidentMatMask.xml";
	storeDispMapToXML(storeName, *m_pConfidentMask);
}

void ConfidenceCompute::confidentCircleJudge(cv::Mat &confidentMat, cv::Mat &mask, const RawImageParameter &rawImageParameter,
	const MicroImageParameter &microImageParameter)
{
//#pragma omp parallel for
	for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
	{
		uchar *yRows = (uchar *)mask.ptr<uchar>(y);
		for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
		{
			if (!confidentCircleJudge(confidentMat, y, x, rawImageParameter, microImageParameter))
				yRows[x] = 255;
		}
	}
}

bool ConfidenceCompute::confidentCircleJudge(cv::Mat &confidentMat, int y, int x, const RawImageParameter &rawImageParameter,
	const MicroImageParameter &microImageParameter)
{
	Point2d &centerPos = microImageParameter.m_ppLensCenterPoints[y][x];
	int curCenterIndex = y*rawImageParameter.m_xLensNum + x;
	double sumCircle = 0.0, countPoints = 0;

	for (int py = centerPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
		py <= centerPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; py++)
	{
		for (int px = centerPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
			px <= centerPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; px++)
		{
			uchar *yConf = confidentMat.ptr<uchar>(py - rawImageParameter.m_yPixelBeginOffset);

			if (microImageParameter.m_ppPixelsMappingSet[py][px] == curCenterIndex){
				sumCircle += yConf[px - rawImageParameter.m_xPixelBeginOffset];
				countPoints += 1.0;
			}
		}
	}

	if (sumCircle / countPoints > MNN_CONFIDENT_MEASURE_THRES)
		return true;
	else
		return false;
}

void ConfidenceCompute::drawConfidentCircle(cv::Mat *&costVol, cv::Mat &mask, const RawImageParameter &rawImageParameter,
	const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter, cv::Mat &srcImg)
{
	cv::Mat rawDisp = cv::Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_8UC1);
	WTAMatch(costVol, rawDisp, disparityParameter.m_disNum);
	double minVal, maxVal;
	minMaxLoc(rawDisp, &minVal, &maxVal);
	cv::Mat disp2;
	rawDisp.convertTo(disp2, CV_8UC1, 255.0 / (maxVal - minVal), -minVal*255.0 / (maxVal - minVal));

	cv::Mat inputImg;
	srcImg.convertTo(inputImg, CV_8UC3);
	//cv::cvtColor(inputImg, inputImg, CV_RGB2BGR);

	std::string rawImgStoreName = m_folderPath + "/ori_circle.png";
	lowTextureAreaPlot(rawImageParameter, microImageParameter, inputImg, mask, rawImgStoreName, CircleDrawMode::e_color, true);

	std::string dispImgStoreName = m_folderPath + "/disp_circle.png";
	lowTextureAreaPlot(rawImageParameter, microImageParameter, disp2, mask, dispImgStoreName, CircleDrawMode::e_gray, true);
}

void ConfidenceCompute::lowTextureAreaPlot(const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter,
	const Mat &img_input, const Mat &mask, const string &picName, CircleDrawMode _circleDrawMode, bool _isOffset)
{
	Mat res_input = img_input.clone();
	int offsetY = _isOffset ? rawImageParameter.m_yPixelBeginOffset : 0;
	int offsetX = _isOffset ? rawImageParameter.m_xPixelBeginOffset : 0;
//#pragma omp parallel for
	for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
	{
		uchar *yRows = (uchar *)mask.ptr<uchar>(y);
		for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
		{
			if (yRows[x] > 125)
			{
				Point2d curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x] - Point2d(offsetX, offsetY);
				drawCircle(res_input, curCenterPos, _circleDrawMode);
			}
		}
	}
	imwrite(picName, res_input);
}

void ConfidenceCompute::drawCircle(Mat &img, const Point2d &centerPos, CircleDrawMode _circleDrawMode)
{
	for (double angle = 0; angle < 360.0; angle += 1.0)
	{
		int index_y = int(centerPos.y - RADIUS*sin(angle / 180.0*PI) + 0.5);
		int index_x = int(centerPos.x + RADIUS*cos(angle / 180.0*PI) + 0.5);

		uchar *ydata = (uchar *)img.ptr<uchar>(index_y);
		if (_circleDrawMode == CircleDrawMode::e_color)
		{
			ydata[3 * index_x] = 0; ydata[3 * index_x + 1] = 0; ydata[3 * index_x + 2] = 255;
		}
		else if (_circleDrawMode == CircleDrawMode::e_gray)
			ydata[index_x] = 255;
	}
}

void ConfidenceCompute::confidentMaskRepair(cv::Mat &confidentMat, const RawImageParameter &rawImageParameter,
	const MicroImageParameter &microImageParameter)
{//置信度图利用梯度圆做修复
	int height = confidentMat.rows;
	int width = confidentMat.cols;

//#pragma omp parallel for
	for (int py = 0; py < height; ++py)
	{
		uchar *pYMask = (uchar *)confidentMat.ptr<uchar>(py);
		for (int px = 0; px < width; ++px)
		{
			int c_index = microImageParameter.m_ppPixelsMappingSet[py][px];
			if (c_index > 0){
				int cy = c_index / rawImageParameter.m_xLensNum;
				int cx = c_index % rawImageParameter.m_xLensNum;

				uchar *ydata = (uchar *)(*m_pGradientCircleMask).ptr<uchar>(cy);
				if (ydata[cx] == 0)
					pYMask[px] = 0;
				//if (m_pGradientCircleMask->at<uchar>(cy, cx) == 0)
					//pYMask[px] = 0;
			}
		}
	}
}