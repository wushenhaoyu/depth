#include "CostVolCompute.h"
#include "DataParameter.cuh"

using namespace std;
using namespace cv;

CostVolCompute::CostVolCompute()
{

}

CostVolCompute::~CostVolCompute()
{

}



void CostVolCompute::costVolDataCompute(const DataParameter &dataParameter, Mat *costVol) //���205�����ݣ�����ֻ���ǳ���������
{//��������Rawͼ���Ӧ��dataCost
	RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
	MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
	DisparityParameter disparityParameter = dataParameter.getDisparityParameter();

	dataParameter.m_inputImg.convertTo(m_inputImg, CV_32FC3, 1 / 255.0f);   //��ͨ����ɫͼ����
	//dataParameter.m_inputImg.convertTo(m_inputImg, CV_32FC1, 1 / 255.0f);	//��ͨ����ɫͼ�����������У�����������������ж�Ҫ�ģ�
	cv::Mat im_gray, tmp;
	m_inputImg.convertTo(tmp, CV_32F);
	cv::cvtColor(tmp, im_gray, COLOR_RGB2GRAY); //���Ƴ��������Ϊ��������Ҫ�����Ҷ�ͼ��Ȼ��������,��Ϊ������ɫ�ݶȷ�����Ҫ�õ���

	cv::Mat dst_x, dst_y;
	/*
	Sobel(im_gray, dst_x, CV_32F, 1, 0);
	Sobel(im_gray, dst_y, CV_32F, 0, 1);
	addWeighted(dst_x, 0.5, dst_y, 0.5, 0, m_gradImg);
	*/
	
	
	
	
	cv::Sobel(im_gray, m_gradImg, CV_32F, 1, 0, 1);
	m_gradImg += 0.5;

	
//#pragma omp parallel for
	for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
	{
		for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
		{
			std::cout << "cost --- y=" << y << "\tx=" << x << std::endl;
			costVolDataCompute(costVol, y, x, rawImageParameter, microImageParameter, disparityParameter);
		}
	}
	
	/*
	int yOffset = rawImageParameter.m_yCenterBeginOffset;
	int xOffset = rawImageParameter.m_xCenterBeginOffset;
	int yLens = rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset - rawImageParameter.m_yCenterBeginOffset;
	int xLens = rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset - rawImageParameter.m_xCenterBeginOffset;
#pragma omp parallel for
	for (int k = 0; k < yLens*xLens; k++)
	{
		int y = k / xLens + yOffset;
		int x = k % xLens + xOffset;
		costVolDataCompute(costVol, y, x, rawImageParameter, microImageParameter, disparityParameter);
	}
	*/
}



void CostVolCompute::costVolDataCompute(cv::Mat *costVol, int y, int x, const RawImageParameter &rawImageParameter,
	const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter)
{//��������Rawͼ���Ӧ��dataCost--����ÿ����ͼ����ÿ�����ص�cost
	Point2d &centerPos = microImageParameter.m_ppLensCenterPoints[y][x];
	int curCenterIndex = y*rawImageParameter.m_xLensNum + x;

	//��������ĵ��Բ��Χ�ڱ�����
	for (int py = centerPos.y - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
		py <= centerPos.y + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; py++)
	{
		for (int px = centerPos.x - microImageParameter.m_circleDiameter / 2 + microImageParameter.m_circleNarrow; 
			px <= centerPos.x + microImageParameter.m_circleDiameter / 2 - microImageParameter.m_circleNarrow; px++)
		{
			if (microImageParameter.m_ppPixelsMappingSet[py][px] == curCenterIndex){//ȷ������Բ�ķ�Χ��
				for (int d = 0; d < disparityParameter.m_disNum; d++)
				{
					float *cost = (float*)costVol[d].ptr<float>(py - rawImageParameter.m_yPixelBeginOffset);
					cost[px - rawImageParameter.m_xPixelBeginOffset] = costVolDataCompute(y, x, py, px, d, rawImageParameter, microImageParameter, disparityParameter);
				}
			}
		}
	}
}

float CostVolCompute::costVolDataCompute(int y, int x, int py, int px, int d, const RawImageParameter &rawImageParameter,
	const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter)
{//��������Rawͼ���Ӧ��dataCost--����ÿ����������Χ�����ص�ƥ��ֵ
	float tempSumCost = 0.0; int tempCostNum = 0;
	Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];
	Point2d matchPoint;
	float realDisp = disparityParameter.m_dispStep*d + disparityParameter.m_dispMin;
	MatchNeighborLens *matchNeighborLens = microImageParameter.m_ppMatchNeighborLens[y][x];

	for (int i = 0; i < NEIGHBOR_MATCH_LENS_NUM; i++)//6������ֱ����
	{
		float matchCenterPos_y = matchNeighborLens[i].m_centerPosY;
		float matchCenterPos_x = matchNeighborLens[i].m_centerPosX;
		float centerDis = matchNeighborLens[i].m_centerDis;
		if (matchCenterPos_y < 0)
			break;

		//����λ������Ҳ��Ԥ���趨�õ�����ݶȣ�ȥ�ӽ�����С�����ص��λ��(ע��˴��ǶԱ�Ե�����ͼ��������)
		matchPoint.y = (centerDis + realDisp)*(matchCenterPos_y - curCenterPos.y) / centerDis + py;
		matchPoint.x = (centerDis + realDisp)*(matchCenterPos_x - curCenterPos.x) / centerDis + px;
		int matchCenterIndex = matchNeighborLens[i].m_centerIndex;

		if (!isCurPointValid(matchPoint, matchCenterIndex, rawImageParameter, microImageParameter))
			continue;
		Point2d curPoint(px, py);
		tempSumCost += bilinearInsertValue(curPoint, matchPoint);
		tempCostNum++;
	}

	if (tempCostNum != 0)
		tempSumCost /= tempCostNum;
	return tempSumCost;
}

float CostVolCompute::bilinearInsertValue(const Point2d &curPoint, const Point2d &matchPoint)
{//�������ĵ���ƥ�����costֵ

	//Դ���RGB��grey
	float* lC = (float *)m_inputImg.ptr<float>(int(curPoint.y)) + 3 * int(curPoint.x); 
	float* lG = (float *)m_gradImg.ptr<float>(int(curPoint.y)) + int(curPoint.x);

	if (int(matchPoint.y) == matchPoint.y && int(matchPoint.x) == matchPoint.x)
	{//����Ҫ��ֵ
		//Ŀ���
		float* rC = (float *)m_inputImg.ptr<float>(int(matchPoint.y)) + 3 * int(matchPoint.x);
		float* rG = (float *)m_gradImg.ptr<float>(int(matchPoint.y)) + int(matchPoint.x);
		return myCostGrd(lC, rC, lG, rG);
	}
	else
	{//˫���Բ�ֵ
		int tempRx = int(matchPoint.x), tempRy = int(matchPoint.y);
		double alphaX = matchPoint.x - tempRx, alphaY = matchPoint.y - tempRy;

		float tempRc[3], tempRg;
		float *rgb_y1 = (float *)m_inputImg.ptr<float>(tempRy);
		float *rgb_y2 = (float *)m_inputImg.ptr<float>(tempRy + 1);
		for (int i = 0; i < 3; i++)
		{
			tempRc[i] = (1 - alphaX)*(1 - alphaY)*rgb_y1[3 * tempRx + i] + alphaX*(1 - alphaY)*rgb_y1[3 * (tempRx + 1) + i] +
				(1 - alphaX)*alphaY*rgb_y2[3 * tempRx + i] + alphaX*alphaY*rgb_y2[3 * (tempRx + 1) + i];
		}

		float *grd_y1 = (float *)m_gradImg.ptr<float>(tempRy);
		float *grd_y2 = (float *)m_gradImg.ptr<float>(tempRy + 1);
		tempRg = (1 - alphaX)*(1 - alphaY)*grd_y1[tempRx] + alphaX*(1 - alphaY)*grd_y1[tempRx + 1] +
			(1 - alphaX)*alphaY*grd_y2[tempRx] + alphaX*alphaY*grd_y2[tempRx + 1];

		float* rC = tempRc;
		float* rG = &tempRg;
		return myCostGrd(lC, rC, lG, rG);
	}
}

bool CostVolCompute::isCurPointValid(Point2d &matchPoint, int matchCenterIndex, const RawImageParameter &rawImageParameter,
		const MicroImageParameter &microImageParameter)
{//�жϴ�λ�õ�ƥ����Ƿ����
	float pm_y = matchPoint.y;
	float pm_x = matchPoint.x;

	if (pm_y < 0 || pm_y >= rawImageParameter.m_srcImgHeight || pm_x < 0 || pm_x >= rawImageParameter.m_srcImgWidth)
		return false;

	if (microImageParameter.m_ppPixelsMappingSet[int(pm_y)][int(pm_x)] != matchCenterIndex)
		return false;

	return true;
}

float CostVolCompute::myCostGrd(float* lC, float* rC, float* lG, float* rG)
{//��ɫ���ݶȽ��м�Ȩ
	float clrDiff = 0;
	// three color
	for (int c = 0; c < 3; c++) {
		float temp = fabs(lC[c] - rC[c]);
		clrDiff += temp;
	}
	clrDiff *= 0.3333333333;
	// gradient diff
	float grdDiff = fabs(lG[0] - rG[0]);
	//clrDiff = clrDiff > TAU_1 ? TAU_1 : clrDiff; //�޸�
	//grdDiff = grdDiff > TAU_2 ? TAU_2 : grdDiff; //�޸�ȡ����������
	//return ALPHA * clrDiff + (1 - ALPHA) * grdDiff;//�޸Ĵ˴����룬����ֻ���ػҶȵļ�����
	return grdDiff;
}

float myCostGrd(float* lC, float* lG)
{//��ɫ���ݶȼ�Ȩ���߽紦��
	float clrDiff = 0;
	// three color
	for (int c = 0; c < 3; c++) {
		float temp = fabs(lC[c] - cv::BORDER_CONSTANT);
		clrDiff += temp;
	}
	clrDiff *= 0.3333333333;
	// gradient diff
	float grdDiff = fabs(lG[0] - cv::BORDER_CONSTANT);
	//clrDiff = clrDiff > TAU_1 ? TAU_1 : clrDiff;
	//grdDiff = grdDiff > TAU_2 ? TAU_2 : grdDiff;
	//return ALPHA * clrDiff + (1 - ALPHA) * grdDiff;
	return grdDiff;
}