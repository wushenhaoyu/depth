/*!
 * \class 
 *
 * \brief �Ӳ��DataCost����ģ�飬������չ��ͬ��Cost���㺯�����иĽ�
 *
 * \author liuqian
 * \date һ�� 2018
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

	void costVolDataCompute(const DataParameter &dataParameter, cv::Mat *costVol); //��������Rawͼ���Ӧ��dataCost
private:
	void costVolDataCompute(cv::Mat *costVol, int y, int x, const RawImageParameter &rawImageParameter,
		const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter);
	//��������Rawͼ���Ӧ��dataCost--����ÿ����ͼ����ÿ�����ص�cost

	float costVolDataCompute(int y, int x, int py, int px, int d, const RawImageParameter &rawImageParameter,
		const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter); 
	//��������Rawͼ���Ӧ��dataCost--����ÿ����������Χ�����ص�ƥ��ֵ

	bool isCurPointValid(const Point2d& matchPoint, int matchCenterIndex, 
                                     const RawImageParameter& rawImageParameter, 
                                     const MicroImageParameter& microImageParameter); //�жϴ�λ�õ�ƥ����Ƿ����

	float bilinearInsertValue(const Point2d &curPoint, const Point2d &matchPoint); //�������ĵ���ƥ�����costֵ

	inline float  myCostGrd(const float* lC, const float* rC, const float* lG, const float* rG); //��ɫ���ݶȽ��м�Ȩ

	inline float myCostGrd(float* lC, float* lG); //��ɫ���ݶȼ�Ȩ���߽紦��

private:
	cv::Mat m_inputImg; //����ԭʼͼ��
	cv::Mat m_gradImg; //ԭʼͼ���Ӧ���ݶ�ͼ��

};


#endif