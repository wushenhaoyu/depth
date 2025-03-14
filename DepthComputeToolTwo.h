/*!
 * \class DepthComputeToolTwo
 *
 * \brief 深度估计模块调用示例，对应方法二
 *
 * \author liuqian
 * \date 一月 2018
 */

#ifndef __DEPTHCOMPUTETOOLTWO_H_
#define __DEPTHCOMPUTETOOLTWO_H_

#include "CommFunc.h"
#include "DataParameter.h"

class DepthComputeToolTwo
{
public:
	DepthComputeToolTwo();
	~DepthComputeToolTwo();
	void parameterInit(std::string dataFolderName, std::string centerPointFileName, std::string inputRawImgName,
		int yCenterBeginOffset = 2, int xCenterBeginOffset = 2, int yCenterEndOffset = 2, int xCenterEndOffset = 2,
		int filterRadius = 4, float circleDiameter = 34.0, float circleNarrow = 1.5, int dispMin = 5, int dispMax = 13, float dispStep = 0.5);
	void rawImageDisparityCompute();
	void sceneDepthCompute(std::string referSubImgName, std::string referDispXmlName, std::string referMaskXmlName, std::string mappingFileName);
private:
	DataParameter m_dataParameter;
};

#endif