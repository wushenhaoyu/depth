/*!
 * \class CostVolFilter
 *
 * \brief ��CostVol�����˲�����
 *
 * \author liuqian
 * \date һ�� 2018
 */

#ifndef __COSTVOLFILTER_H_
#define __COSTVOLFILTER_H_

#include "CommFunc.h"

class DataParameter;
struct RawImageParameter;
struct MicroImageParameter;
struct DisparityParameter;
struct FilterPatameter;

class CostVolFilter
{
public:
	void costVolWindowFilter(const DataParameter &dataParameter, cv::Mat *costVol, cv::Mat *costVolFilter);//��costVol���д����˲�
	void costVolWindowFilter(const DataParameter &dataParameter, cv::Mat *costVol);
	//void microImageDisparityFilter(const DataParameter &dataParameter, cv::Mat *&costVol, FilterOptimizeKind curFilterOptimizeKind);//��С��microͼ������˲�
	//void gpuFilterTest(const DataParameter &dataParameter, cv::Mat *&costVol);
private:
	void costVolBoundaryRepair(cv::Mat *costVol, const DisparityParameter &disparityParameter, const FilterPatameter &filterPatameter);//��Բ�α߽���д���
	void costVolWindowFilter(cv::Mat *costVol, cv::Mat *costVolFilter, int y, int x, const RawImageParameter &rawImageParameter,
		const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter, const FilterPatameter &filterPatameter);//��costVol���д����˲�
	void costVolWindowFilter(cv::Mat *costVol, int y, int x, const RawImageParameter &rawImageParameter,
		const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter, const FilterPatameter &filterPatameter);//��costVol���д����˲�
};

#endif // !__COSTVOLFILTER_H_