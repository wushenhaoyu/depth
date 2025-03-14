/*!
 * \class VirtualDepthCompute
 *
 * \brief �����������
 *
 * \author liuqian
 * \date һ�� 2018
 */

#ifndef __VIRTUALDEPTHCOMPUTE_H_
#define __VIRTUALDEPTHCOMPUTE_H_

#include "CommFunc.h"

class DataParameter;

class VirtualDepthCompute
{
public:
	VirtualDepthCompute();
	~VirtualDepthCompute();
	void virtualDepthCompute(const DataParameter &dataParameter, cv::Mat &referDispMap, cv::Mat &referDiskMask); //�����������
private:
	//void virtualDepthPointStore(std::string pathName, std::string storeFileName);//����ά�������
	void virtualDepthImageCreat(std::string pathName, std::string storeImgName);//�����������ͼ��
	vector<Point3f> m_pVirtualPointVec;//������ά�㼯��

};



#endif