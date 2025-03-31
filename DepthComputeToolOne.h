/*!
 * \class DepthComputeToolOne
 *
 * \brief ��ȼ�����ò�ͬģ��ʾ����ʾ����Ӧ����һ
 *
 * \author liuqian
 * \date һ�� 2018
 */

#ifndef _DEPTHCOMPUTETOOLONE_H_
#define _DEPTHCOMPUTETOOLONE_H_

#include "CommFunc.h"
#include "DataParameter.cuh"

class DepthComputeToolOne
{
public:
	DepthComputeToolOne();
	~DepthComputeToolOne();

	void parameterInit(std::string dataFolderName, std::string centerPointFileName, std::string inputRawImgName,
		int yCenterBeginOffset = 2, int xCenterBeginOffset = 2, int yCenterEndOffset = 2, int xCenterEndOffset = 2,
		int filterRadius = 4, float circleDiameter = 34.0, float circleNarrow = 1.5, int dispMin = 5, int dispMax = 13, float dispStep = 0.5);
	void rawImageDisparityCompute();
	void sceneDepthCompute(std::string subApertureImgName, std::string mappingFileName);
private:
	DataParameter m_dataParameter;
};


#endif