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
	clock_t t1 = clock();
	m_dataParameter.init(dataFolderName, centerPointFileName, inputRawImgName,
		yCenterBeginOffset, xCenterBeginOffset, yCenterEndOffset, xCenterEndOffset,
		filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);
	clock_t t2 = clock();
	std::cout << "init parameter use  time: " << (t2 - t1)  << " ms " << std::endl;
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
	/*cv::cuda::GpuMat* costVol = new cv::cuda::GpuMat[disparityParameter.m_disNum];
    for (int mIdx = 0; mIdx < disparityParameter.m_disNum; mIdx++) {
        costVol[mIdx] = cv::cuda::GpuMat(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_32FC1);
        costVol[mIdx].setTo(cv::Scalar(0));  // 在 GPU 上初始化为零
    }*/
	fstream outfile(m_dataParameter.m_folderPath + "/timeLog.txt", ios::out);

//	/*
	CostVolCompute costVolCompute;
	t1 = clock();
	costVolCompute.costVolDataCompute(m_dataParameter, costVol);
	t2 = clock();
	std::cout << "init raw cost vol compute use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds " << std::endl;
//	*/

	DataDeal dataDeal;
	std::string  storeName;
	cv::Mat rawDisp = cv::Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_8UC1);
	//rawDisp.setTo(5);//��ʱ�趨Ϊ�̶�ֵ��ȡ������ļ������
	dataDeal.WTAMatch(costVol, rawDisp, disparityParameter.m_disNum);//����
	storeName = m_dataParameter.m_folderPath + "/dispBeforeFilter.png";
	dataDeal.dispMapShow(storeName, rawDisp);//����
	std::cout << "disp before filter final!" << std::endl;


	

	CostVolFilter costVolFilter;
	t1 = clock();
	costVolFilter.costVolWindowFilter(m_dataParameter, costVol);
	t2 = clock();
	//std::cout << "init raw cost vol filter use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds " << std::endl;
	std::cout << "init raw cost vol filter use time: " << (t2 - t1)  << " ms " << std::endl;
	dataDeal.WTAMatch(costVol, rawDisp, disparityParameter.m_disNum);
	storeName = m_dataParameter.m_folderPath + "/dispAfterLocalSmooth.png";
	dataDeal.dispMapShow(storeName, rawDisp);//����
	std::cout << "disp_afterFilter final!" << std::endl;
	



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
	t1 = clock();
	confidenceCompute.confidenceMeasureCompute(m_dataParameter, costVol);
	t2 = clock();
	std::cout << "confident measure compute use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds " << std::endl;
	outfile << "confident measure compute use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds \n";
	cv::Mat *pConfidentMask = confidenceCompute.getConfidentMask();
	

	SceneDepthCompute sceneDepthCompute;
	sceneDepthCompute.outputMicrolensDisp(m_dataParameter, rawDisp, pConfidentMask);
//	*/

	ImageRander imageRander;
	t1 = clock();
	imageRander.imageRanderWithMask(m_dataParameter, rawDisp, pConfidentMask);
//	imageRander.imageRanderWithOutMask(m_dataParameter, rawDisp);
	t2 = clock();
	std::cout << "sub aperature rander use time: " << (t2 - t1)  << " ms " << std::endl;
	outfile << "sub aperature rander use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds \n";
	
	outfile.close();
	delete[]costVol;
}

void DepthComputeToolTwo::sceneDepthCompute(std::string referSubImgName, std::string referDispXmlName, std::string referMaskXmlName, std::string mappingFileName)
{
	SceneDepthCompute sceneDepthCompute;
	sceneDepthCompute.loadSceneDataCost(m_dataParameter, referSubImgName, referDispXmlName, referMaskXmlName, mappingFileName);
}