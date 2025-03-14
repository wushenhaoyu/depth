/*!
 * \class SceneDepthCompute
 *
 * \brief 计算场景深度
 *
 * \author liuqian
 * \date 一月 2018
 */

#ifndef __SCENEDEPTHCOMPUTE_H_
#define __SCENEDEPTHCOMPUTE_H_

#include "CommFunc.h"
#include "DataDeal.h"

class DataParameter;
struct RawImageParameter;
struct MicroImageParameter;
struct DisparityParameter;

class SceneDepthCompute : public DataDeal
{
public:
	SceneDepthCompute();
	~SceneDepthCompute();
	void loadSceneDataCost(const DataParameter &dataParameter, std::string referImgName, std::string referDispXmlName,
		std::string referMaskXmlName, std::string mappingFileName);//新的对应视差的载入
	void loadSceneDataCost(const DataParameter &dataParameter, cv::Mat &subApertureImg, std::string mappingFileName, 
		cv::Mat *srcCostVol, cv::Mat *destCostVol);//新的对应视差的载入
	void outputMicrolensDisp(const DataParameter &dataParameter, Mat &rawDisp, Mat *confidentMask = nullptr); //输出微透镜图像中平均视差文件
private:
	float getSmoothValue(const RawImageParameter &rawImageParameter, const DisparityParameter &disparityParameter, Mat &rawDisp,
		Point2d &centerPos, Mat *confidentMask = nullptr, const int shift_size = 10);
	void fillOtherCostVol(int height, int width, int maxDis, Mat* &costVol);//对一些没有cost的值就行填补
};

#endif