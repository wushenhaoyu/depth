/*!
 * \class 
 *
 * \brief 视差的DataCost计算模块，可以扩展不同的Cost计算函数进行改进
 *
 * \author liuqian
 * \date 一月 2018
 */

#ifndef __COSTVOLCOMPUTE_H_
#define __COSTVOLCOMPUTE_H_

#include "CommFunc.h"

class DataParameter;
struct RawImageParameter;
struct MicroImageParameter;
struct DisparityParameter;

class CostVolCompute
{
public:
	CostVolCompute();
	~CostVolCompute();

	void costVolDataCompute(const DataParameter &dataParameter, cv::Mat *costVol); //用来计算Raw图像对应的dataCost
private:
	void costVolDataCompute(cv::Mat *costVol, int y, int x, const RawImageParameter &rawImageParameter,
		const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter);
	//用来计算Raw图像对应的dataCost--计算每个子图像中每个像素的cost

	float costVolDataCompute(int y, int x, int py, int px, int d, const RawImageParameter &rawImageParameter,
		const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter); 
	//用来计算Raw图像对应的dataCost--计算每个像素与周围的像素的匹配值

	bool isCurPointValid(Point2d &matchPoint, int matchCenterIndex, const RawImageParameter &rawImageParameter,
		const MicroImageParameter &microImageParameter); //判断此位置的匹配点是否合理

	float bilinearInsertValue(const Point2d &curPoint, const Point2d &matchPoint); //计算中心点与匹配点间的cost值

	inline float myCostGrd(float* lC, float* rC, float* lG, float* rG); //颜色与梯度进行加权

	inline float myCostGrd(float* lC, float* lG); //颜色与梯度加权，边界处理

private:
	cv::Mat m_inputImg; //输入原始图像
	cv::Mat m_gradImg; //原始图像对应的梯度图像

};


#endif