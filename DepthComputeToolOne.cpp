#include "DepthComputeToolOne.h"
#include "CostVolCompute.h"
#include "DataDeal.h"
#include "CostVolFilter.h"
#include "SceneDepthCompute.h"
#include "DisparityRefinement.h"
#include "ConfidenceCompute.h"
#include "SceneDepthCompute.h"
#include <fstream>
#include <time.h>

DepthComputeToolOne::DepthComputeToolOne()
{

}

DepthComputeToolOne::~DepthComputeToolOne()
{

}

void DepthComputeToolOne::parameterInit(std::string dataFolderName, std::string centerPointFileName, std::string inputRawImgName,
	int yCenterBeginOffset, int xCenterBeginOffset, int yCenterEndOffset, int xCenterEndOffset,
	int filterRadius, float circleDiameter, float circleNarrow, int dispMin, int dispMax, float dispStep)
{
	clock_t t1 = clock();
	m_dataParameter.init(dataFolderName, centerPointFileName, inputRawImgName,
		yCenterBeginOffset, xCenterBeginOffset, yCenterEndOffset, xCenterEndOffset,
		filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);
	clock_t t2 = clock();
	std::cout << "init parameter use  time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds " << std::endl;
}

void DepthComputeToolOne::rawImageDisparityCompute()
{
	RawImageParameter rawImageParameter = m_dataParameter.getRawImageParameter();
	DisparityParameter disparityParameter = m_dataParameter.getDisparityParameter();
	clock_t t1, t2;

	fstream outfile(m_dataParameter.m_folderPath + "/timeLog.txt", ios::out);

	cv::Mat *costVol = new cv::Mat[disparityParameter.m_disNum];
	for (int mIdx = 0; mIdx < disparityParameter.m_disNum; mIdx++) {
		costVol[mIdx] = cv::Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_32FC1);
	}

	CostVolCompute costVolCompute;
	t1 = clock();
	costVolCompute.costVolDataCompute(m_dataParameter, costVol);
	t2 = clock();
	std::cout << "init raw cost vol compute use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds " << std::endl;
	outfile << "init raw cost vol compute use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds \n";

	DataDeal dataDeal;
	std::string  storeName;
	cv::Mat rawDisp = cv::Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_8UC1);
	dataDeal.WTAMatch(costVol, rawDisp, disparityParameter.m_disNum);
	storeName = m_dataParameter.m_folderPath + "/dispBeforeFilter.png";
	dataDeal.dispMapShow(storeName, rawDisp);
	std::cout << "disp before filter final!" << std::endl;

	CostVolFilter costVolFilter;
	t1 = clock();
	costVolFilter.costVolWindowFilter(m_dataParameter, costVol);
	//costVolFilter.gpuFilterTest(m_dataParameter, costVol);
	t2 = clock();
	std::cout << "init raw cost vol filter use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds " << std::endl;
	outfile << "init raw cost vol filter use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds \n";
	dataDeal.WTAMatch(costVol, rawDisp, disparityParameter.m_disNum);
	storeName = m_dataParameter.m_folderPath + "/dispAfterLocalSmooth.png";
	dataDeal.dispMapShow(storeName, rawDisp);
	std::cout << "disp_afterFilter final!" << std::endl;

	t1 = clock();
	//costVolFilter.microImageDisparityFilter(m_dataParameter, costVol, FilterOptimizeKind::e_stca); 
	t2 = clock();
	std::cout << "raw cost micro image disparity refinement use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds " << std::endl;
	outfile << "raw cost micro image disparity refinement use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds \n";
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

	SceneDepthCompute sceneDepthCompute;
	sceneDepthCompute.outputMicrolensDisp(m_dataParameter, rawDisp, nullptr);

	outfile.close();
	delete[]costVol;
}

void DepthComputeToolOne::sceneDepthCompute(std::string subApertureImgName, std::string mappingFileName)
{
	string storeName;
	Mat sceneImg = imread(m_dataParameter.m_folderPath + "/" + subApertureImgName, IMREAD_COLOR);
	int sceneImg_height = sceneImg.rows, sceneImg_width = sceneImg.cols;
	int maxDis = m_dataParameter.getDisparityParameter().m_disNum;
	RawImageParameter rawImageParameter = m_dataParameter.getRawImageParameter();
	Mat sceneDis = Mat::zeros(sceneImg_height, sceneImg_width, CV_8UC1);

	cv::Mat *srcCostVol = new cv::Mat[maxDis];
	for (int mIdx = 0; mIdx < maxDis; mIdx++) {
		srcCostVol[mIdx] = cv::Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_32FC1);
	}

	CostVolCompute costVolCompute;
	costVolCompute.costVolDataCompute(m_dataParameter, srcCostVol);
	CostVolFilter costVolFilter;
	costVolFilter.costVolWindowFilter(m_dataParameter, srcCostVol);
	//costVolFilter.microImageDisparityFilter(m_dataParameter, srcCostVol, FilterOptimizeKind::e_stca);
	costVolFilter.costVolWindowFilter(m_dataParameter, srcCostVol);

	Mat *sceneCostVol = new Mat[maxDis];
	for (int mIdx = 0; mIdx < maxDis; mIdx++) {
		sceneCostVol[mIdx] = Mat::zeros(sceneImg_height, sceneImg_width, CV_32FC1);
	}

	SceneDepthCompute sceneDepthCompute;
	sceneDepthCompute.loadSceneDataCost(m_dataParameter, sceneImg, mappingFileName, srcCostVol, sceneCostVol);
	
	DataDeal dataDeal;
	dataDeal.WTAMatch(sceneCostVol, sceneDis, maxDis);
	storeName = m_dataParameter.m_folderPath + "/disp_beforeGlobalGCOptimize.png";
	dataDeal.dispMapShow(storeName, sceneDis);
	storeName = m_dataParameter.m_folderPath + "/disp_beforeGlobalGCOptimizeWithColor.png";
	dataDeal.dispMapShowForColor(storeName, sceneDis);
	std::cout << "before GC global optimize!" << std::endl;

	clock_t t1 = clock();
	//DisparityRefinement::getInstance()->globalGCOptimize(sceneImg, maxDis, sceneCostVol, sceneDis);
	clock_t t2 = clock();
	std::cout << "GC global optimize use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds " << std::endl;

	storeName = m_dataParameter.m_folderPath + "/disp_afterGlobalGCOptimize.png";
	dataDeal.dispMapShow(storeName, sceneDis);
	storeName = m_dataParameter.m_folderPath + "/disp_afterGlobalGCOptimizeWithColor.png";
	dataDeal.dispMapShowForColor(storeName, sceneDis);
	std::cout << "before GC global optimize!" << std::endl;
	storeName = m_dataParameter.m_folderPath + "/disp_afterGlobalGCOptimize.xml";
	dataDeal.storeDispMapToXML(storeName, sceneDis);

	delete[]srcCostVol;
	delete[]sceneCostVol;
}