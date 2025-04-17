#include "DepthComputeToolTwo.h"
#include "CostVolCompute.h"
#include "DataDeal.h"
#include "CostVolFilter.h"
#include "SceneDepthCompute.h"
#include "DisparityRefinement.h"
#include "ConfidenceCompute.h"
#include "SceneDepthCompute.h"
#include "ImageRander.h"
#include <time.h>

DepthComputeToolTwo::DepthComputeToolTwo()
{

}

DepthComputeToolTwo::~DepthComputeToolTwo()
{

}

void DepthComputeToolTwo::parameterInit(std::string dataFolderName, std::string centerPointFileName, std::string inputRawImgName,
	int yCenterBeginOffset, int xCenterBeginOffset, int yCenterEndOffset, int xCenterEndOffset, int filterRadius, float circleDiameter, 
	float circleNarrow, int dispMin, int dispMax, float dispStep)
{
	auto start = std::chrono::high_resolution_clock::now();
	m_dataParameter.init(dataFolderName, centerPointFileName, inputRawImgName,
		yCenterBeginOffset, xCenterBeginOffset, yCenterEndOffset, xCenterEndOffset,
		filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "init parameter use  time: " << duration << " ms " << std::endl;
}

void DepthComputeToolTwo::rawImageDisparityCompute()
{
	RawImageParameter rawImageParameter = m_dataParameter.getRawImageParameter();
	DisparityParameter disparityParameter = m_dataParameter.getDisparityParameter();


//	/*
	CostVolCompute costVolCompute;
	auto start = std::chrono::high_resolution_clock::now();
	costVolCompute.costVolDataCompute(m_dataParameter, nullptr);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "init raw cost vol compute use time: " << duration  << " ms " << std::endl;
//	*/
 

	start = std::chrono::high_resolution_clock::now();
	DataDeal dataDeal;
	std::string  storeName;
	dataDeal.WTAMatch(rawImageParameter.m_recImgWidth, rawImageParameter.m_recImgHeight, disparityParameter.m_disNum);//����
	std::cout << "disp before filter final!" << std::endl;
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "DataDeal use time: " << duration  << " ms " << std::endl;

	//saveSingleChannelGpuMemoryAsImage(d_rawDisp, rawImageParameter.m_recImgWidth , rawImageParameter.m_recImgHeight, "./res/dispBeforeFilter.png");

	

	CostVolFilter costVolFilter;
	start = std::chrono::high_resolution_clock::now();
	costVolFilter.costVolWindowFilter(m_dataParameter, nullptr);
	dataDeal.WTAMatch(rawImageParameter.m_recImgWidth, rawImageParameter.m_recImgHeight, disparityParameter.m_disNum);//����
	//saveSingleChannelGpuMemoryAsImage(d_rawDisp, rawImageParameter.m_recImgWidth , rawImageParameter.m_recImgHeight, "./res/dispAfterLocalSmooth.png");
	//storeName = m_dataParameter.m_folderPath + "/dispAfterLocalSmooth.png";
	std::cout << "disp_afterFilter final!" << std::endl;
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "init raw cost vol filter use time: " << duration << " ms " << std::endl;

	ImageRander imageRander;
	start = std::chrono::high_resolution_clock::now();
	imageRander.imageRanderWithOutMask(m_dataParameter);
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "sub aperature rander use time: " << duration << " ms " << std::endl;
	

}

void DepthComputeToolTwo::sceneDepthCompute(std::string referSubImgName, std::string referDispXmlName, std::string referMaskXmlName, std::string mappingFileName)
{
	SceneDepthCompute sceneDepthCompute;
	sceneDepthCompute.loadSceneDataCost(m_dataParameter, referSubImgName, referDispXmlName, referMaskXmlName, mappingFileName);
}