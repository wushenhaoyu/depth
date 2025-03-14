/*!
 * \class SceneDepthCompute
 *
 * \brief ���㳡�����
 *
 * \author liuqian
 * \date һ�� 2018
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
		std::string referMaskXmlName, std::string mappingFileName);//�µĶ�Ӧ�Ӳ������
	void loadSceneDataCost(const DataParameter &dataParameter, cv::Mat &subApertureImg, std::string mappingFileName, 
		cv::Mat *srcCostVol, cv::Mat *destCostVol);//�µĶ�Ӧ�Ӳ������
	void outputMicrolensDisp(const DataParameter &dataParameter, Mat &rawDisp, Mat *confidentMask = nullptr); //���΢͸��ͼ����ƽ���Ӳ��ļ�
private:
	float getSmoothValue(const RawImageParameter &rawImageParameter, const DisparityParameter &disparityParameter, Mat &rawDisp,
		Point2d &centerPos, Mat *confidentMask = nullptr, const int shift_size = 10);
	void fillOtherCostVol(int height, int width, int maxDis, Mat* &costVol);//��һЩû��cost��ֵ�����
};

#endif