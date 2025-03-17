/*!
 * \class ConfidenceCompute
 *
 * \brief ���Ŷ�ָ��ļ���
 *
 * \author liuqian
 * \date һ�� 2018
 */

#ifndef __CONFIDENCECOMPUTE_H__
#define __CONFIDENCECOMPUTE_H__

#include "CommFunc.h"
#include "DataDeal.h"

class DataParameter;
struct RawImageParameter;
struct MicroImageParameter;
struct DisparityParameter;
struct FilterPatameter;

class ConfidenceCompute : public DataDeal
{
public:
	enum class CircleDrawMode{
		e_color, e_gray
	};

	ConfidenceCompute();
	~ConfidenceCompute();
	void confidenceMeasureCompute(const DataParameter &dataParameter, cv::Mat *&costVol); //���Ŷȼ��㿪ʼ
	cv::Mat *getConfidentMask() const
	{//��ȡ���Ŷ�mask
		return m_pConfidentMask;
	};
private:
	void gradientMeasureCompute(const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, 
		const FilterPatameter &filterPatameter, cv::Mat &srcImg);//�ݶ�ָ�����
	void confidentMeasureMMN(cv::Mat *&costVol, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, 
		const DisparityParameter &disparityParameter, cv::Mat &srcImg);//����MMN�����Ŷ�ָ�����
	void setConfidentMask(cv::Mat &confidentMat, std::string confidentMaskName, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter);//�������Ŷȵ�mask
	void confidentCircleJudge(cv::Mat &confidentMat, cv::Mat &mask, const RawImageParameter &rawImageParameter, 
		const MicroImageParameter &microImageParameter);
	bool confidentCircleJudge(cv::Mat &confidentMat, int y, int x, const RawImageParameter &rawImageParameter, 
		const MicroImageParameter &microImageParameter);
	void drawConfidentCircle(cv::Mat *&costVol, cv::Mat &mask, const RawImageParameter &rawImageParameter, 
		const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter, cv::Mat &srcImg);
	void lowTextureAreaPlot(const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, 
		const Mat &img_input, const Mat &mask, const string &picName, CircleDrawMode _circleDrawMode, bool _isOffset = false);
	void drawCircle(Mat &img, const Point2d &centerPos, CircleDrawMode _circleDrawMode);
	void confidentMaskRepair(cv::Mat &confidentMat, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter);//���Ŷ�ͼ�����ݶ�Բ���޸�
	
	string m_folderPath; //�洢·��
	cv::Mat *m_pGradientCircleMask; //�ݶ�mask
	cv::Mat *m_pConfidentMask; //���Ŷ�mask
};

#endif