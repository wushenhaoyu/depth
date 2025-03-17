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
struct CudaPoint2f ;
class CostVolCompute
{
public:
	CostVolCompute();
	~CostVolCompute();

	void costVolDataCompute(const DataParameter& dataParameter, Mat* costVol);

private:
	cv::Mat m_inputImg; //����ԭʼͼ��
	cv::Mat m_gradImg; //ԭʼͼ���Ӧ���ݶ�ͼ��

};


#endif