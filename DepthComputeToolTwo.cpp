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
	clock_t t1, t2;

	cv::Mat *costVol = new cv::Mat[disparityParameter.m_disNum];
	for (int mIdx = 0; mIdx < disparityParameter.m_disNum; mIdx++) {
		costVol[mIdx] = cv::Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_32FC1);
	}
	fstream outfile(m_dataParameter.m_folderPath + "/timeLog.txt", ios::out);

//	/*
	CostVolCompute costVolCompute;
	auto start = std::chrono::high_resolution_clock::now();
	costVolCompute.costVolDataCompute(m_dataParameter, costVol);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "init raw cost vol compute use time: " << duration  << " ms " << std::endl;
//	*/


	start = std::chrono::high_resolution_clock::now();
	DataDeal dataDeal;
	std::string  storeName;
	cv::Mat rawDisp = cv::Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_8UC1);
	//rawDisp.setTo(5);//��ʱ�趨Ϊ�̶�ֵ��ȡ������ļ������
	dataDeal.WTAMatch(costVol, rawDisp, disparityParameter.m_disNum);//����
	storeName = m_dataParameter.m_folderPath + "/dispBeforeFilter.png";
	dataDeal.dispMapShow(storeName, rawDisp);//����
	std::cout << "disp before filter final!" << std::endl;
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "DataDeal use time: " << duration  << " ms " << std::endl;

	

	CostVolFilter costVolFilter;
	start = std::chrono::high_resolution_clock::now();
	costVolFilter.costVolWindowFilter(m_dataParameter, costVol);
	//std::cout << "init raw cost vol filter use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds " << std::endl;
	dataDeal.WTAMatch(costVol, rawDisp, disparityParameter.m_disNum);
	storeName = m_dataParameter.m_folderPath + "/dispAfterLocalSmooth.png";
	dataDeal.dispMapShow(storeName, rawDisp);//����
	std::cout << "disp_afterFilter final!" << std::endl;
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "init raw cost vol filter use time: " << duration << " ms " << std::endl;



/*���Ա�������������ò���
	
	t1 = clock();
	costVolFilter.microImageDisparityFilter(m_dataParameter, costVol, FilterOptimizeKind::e_stca);
	t2 = clock();
	std::cout << "raw cost micro image disparity refinement use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds " << std::endl;
	dataDeal.WTAMatch(costVol, rawDisp, disparityParameter.m_disNum);
	storeName = m_dataParameter.m_folderPath + "/dispAfterSTCA.png";
	dataDeal.dispMapShow(storeName, rawDisp);
	std::cout << "disp_afterSTCA final!" << std::endl;
	
	
	
	costVolFilter.costVolWindowFilter(m_dataParameter, costVol);
	dataDeal.WTAMatch(costVol, rawDisp, disparityParameter.m_disNum);
	storeName = m_dataParameter.m_folderPath + "/dispAfterSTCAAgainLocalSmooth.png";
	dataDeal.dispMapShow(storeName, rawDisp);
	storeName = m_dataParameter.m_folderPath + "/dispAfterSTCAAgainLocalSmooth.xml";
	dataDeal.storeDispMapToXML(storeName, rawDisp);
	std::cout << "disp_afterSTCA again smooth final!" << std::endl;
	
*/

//	/*����
	ConfidenceCompute confidenceCompute;
	start = std::chrono::high_resolution_clock::now();
	confidenceCompute.confidenceMeasureCompute(m_dataParameter, costVol);
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "confident measure compute use time: " << duration << " ms" << std::endl;
	outfile << "confident measure compute use time: " << duration << " ms \n";
	cv::Mat *pConfidentMask = confidenceCompute.getConfidentMask();
	
	start = std::chrono::high_resolution_clock::now();
	SceneDepthCompute sceneDepthCompute;
	sceneDepthCompute.outputMicrolensDisp(m_dataParameter, rawDisp, pConfidentMask);
//	*/end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "sence depth compute use time: " << duration << " ms" << std::endl;

	ImageRander imageRander;
	start = std::chrono::high_resolution_clock::now();
	imageRander.imageRanderWithMask(m_dataParameter, rawDisp, pConfidentMask);
//	imageRander.imageRanderWithOutMask(m_dataParameter, rawDisp);
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "sub aperature rander use time: " << duration << " ms " << std::endl;
	outfile << "sub aperature rander use time: " << duration / CLOCKS_PER_SEC << " seconds \n";
	
	outfile.close();
	delete[]costVol;
}

void DepthComputeToolTwo::sceneDepthCompute(std::string referSubImgName, std::string referDispXmlName, std::string referMaskXmlName, std::string mappingFileName)
{
	SceneDepthCompute sceneDepthCompute;
	sceneDepthCompute.loadSceneDataCost(m_dataParameter, referSubImgName, referDispXmlName, referMaskXmlName, mappingFileName);
}