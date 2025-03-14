/*!
 * \class ConfidenceCompute
 *
 * \brief 置信度指标的计算
 *
 * \author liuqian
 * \date 一月 2018
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
	void confidenceMeasureCompute(const DataParameter &dataParameter, cv::Mat *&costVol); //置信度计算开始
	cv::Mat *getConfidentMask() const
	{//获取置信度mask
		return m_pConfidentMask;
	};
private:
	void gradientMeasureCompute(const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, 
		const FilterPatameter &filterPatameter, cv::Mat &srcImg);//梯度指标计算
	void gradientCircleSign(const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, cv::Mat &srcImg);//梯度圆mask标记
	void confidentMeasureMMN(cv::Mat *&costVol, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, 
		const DisparityParameter &disparityParameter, cv::Mat &srcImg);//根据MMN的置信度指标计算
	void confidentMeasureMMN(cv::Mat *&costVol, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter,
		const DisparityParameter &disparityParameter,
		cv::Mat &confMeasureMat, int y, int x, cv::Mat &confidentMat2);
	void setConfidentMask(cv::Mat &confidentMat, std::string confidentMaskName, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter);//设置置信度的mask
	void confidentCircleJudge(cv::Mat &confidentMat, cv::Mat &mask, const RawImageParameter &rawImageParameter, 
		const MicroImageParameter &microImageParameter);
	bool confidentCircleJudge(cv::Mat &confidentMat, int y, int x, const RawImageParameter &rawImageParameter, 
		const MicroImageParameter &microImageParameter);
	void drawConfidentCircle(cv::Mat *&costVol, cv::Mat &mask, const RawImageParameter &rawImageParameter, 
		const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter, cv::Mat &srcImg);
	void lowTextureAreaPlot(const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, 
		const Mat &img_input, const Mat &mask, const string &picName, CircleDrawMode _circleDrawMode, bool _isOffset = false);
	void drawCircle(Mat &img, const Point2d &centerPos, CircleDrawMode _circleDrawMode);
	void confidentMaskRepair(cv::Mat &confidentMat, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter);//置信度图利用梯度圆做修复
	
	string m_folderPath; //存储路径
	cv::Mat *m_pGradientCircleMask; //梯度mask
	cv::Mat *m_pConfidentMask; //置信度mask
};

#endif