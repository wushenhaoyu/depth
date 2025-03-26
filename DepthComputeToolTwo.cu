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
	fstream outfile(m_dataParameter.m_folderPath + "/timeLog.txt", ios::out);

	DataDeal dataDeal;
	std::string  storeName;
	cv::Mat rawDisp = cv::Mat::zeros(rawImageParameter.m_recImgHeight, rawImageParameter.m_recImgWidth, CV_8UC1);
	/*设为固定值，取消计算*/
	for (int i = 0; i < rawDisp.rows; i++) {
		for (int j = 0; j < rawDisp.cols; j++) {
			rawDisp.at<uchar>(i, j) = 15;
		}
	}

	dataDeal.WTAMatch(costVol, rawDisp, disparityParameter.m_disNum);//锟斤拷锟斤拷
	storeName = m_dataParameter.m_folderPath + "/dispBeforeFilter.png";
	//dataDeal.dispMapShow(storeName, rawDisp);//锟斤拷锟斤拷
	std::cout << "disp before filter final!" << std::endl;


	

	CostVolFilter costVolFilter;
	auto start = std::chrono::high_resolution_clock::now();
	costVolFilter.costVolWindowFilter(m_dataParameter, costVol);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	//std::cout << "init raw cost vol filter use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds " << std::endl;
	std::cout << "init raw cost vol filter use time: " << duration  << " ms " << std::endl;
	//dataDeal.WTAMatch(costVol, rawDisp, disparityParameter.m_disNum);
	//storeName = m_dataParameter.m_folderPath + "/dispAfterLocalSmooth.png";
	//dataDeal.dispMapShow(storeName, rawDisp);//锟斤拷锟斤拷
	//std::cout << "disp_afterFilter final!" << std::endl;
	



/*锟斤拷锟皆憋拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷貌锟斤拷锟�
	
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

	/*锟斤拷锟斤拷
	ConfidenceCompute confidenceCompute;
	t1 = clock();
	confidenceCompute.confidenceMeasureCompute(m_dataParameter, costVol);
	t2 = clock();
	std::cout << "confident measure compute use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds " << std::endl;
	outfile << "confident measure compute use time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds \n";
	cv::Mat *pConfidentMask = confidenceCompute.getConfidentMask();
	

	SceneDepthCompute sceneDepthCompute;
	sceneDepthCompute.outputMicrolensDisp(m_dataParameter, rawDisp, pConfidentMask);
	*/

	ImageRander imageRander;
	start = std::chrono::high_resolution_clock::now();
//	imageRander.imageRanderWithMask(m_dataParameter, rawDisp, pConfidentMask);
	imageRander.imageRanderWithOutMask(m_dataParameter, rawDisp);
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "sub aperature rander use time: " << duration  << " ms " << std::endl;
	outfile << "sub aperature rander use time: " << duration / CLOCKS_PER_SEC << " seconds \n";
	
	outfile.close();
	delete[]costVol;
}

/*void DepthComputeToolTwo::sceneDepthCompute(std::string referSubImgName, std::string referDispXmlName, std::string referMaskXmlName, std::string mappingFileName)
{
	SceneDepthCompute sceneDepthCompute;
	sceneDepthCompute.loadSceneDataCost(m_dataParameter, referSubImgName, referDispXmlName, referMaskXmlName, mappingFileName);
}*/