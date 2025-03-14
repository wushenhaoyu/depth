/*!
 * \class VirtualDepthCompute
 *
 * \brief 计算虚拟深度
 *
 * \author liuqian
 * \date 一月 2018
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
	void virtualDepthCompute(const DataParameter &dataParameter, cv::Mat &referDispMap, cv::Mat &referDiskMask); //计算虚拟深度
private:
	//void virtualDepthPointStore(std::string pathName, std::string storeFileName);//将三维点存起来
	void virtualDepthImageCreat(std::string pathName, std::string storeImgName);//生成虚拟深度图像
	vector<Point3f> m_pVirtualPointVec;//虚拟三维点集合

};



#endif