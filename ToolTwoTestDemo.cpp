// Software Version: V1.00
// SoftWare Name:锟斤拷锟斤拷锟斤拷示锟斤拷锟斤拷TS-RJ4001
// Author: Zhang Hongyu
// Date: 2024-12-03





#include "ToolTwoTestDemo.h"
#include "DepthComputeToolTwo.h"

using namespace std;
using namespace cv;

 //#define SCENE_DEPTH_COMPUTE 1

ToolTwoTestDemo::ToolTwoTestDemo()
{
}

ToolTwoTestDemo::~ToolTwoTestDemo()
{
}

void ToolTwoTestDemo::data1compute()
{//锟斤拷锟斤拷0321_2
	//锟斤拷一锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟斤拷锟斤拷锟截诧拷锟斤拷锟斤拷锟斤拷锟斤拷
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x锟斤拷y锟斤拷锟斤拷锟较碉拷图锟斤拷偏锟斤拷值
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; //锟斤拷取锟斤拷图锟斤拷锟斤拷群透叨锟�

	int filterRadius = 6; //锟剿诧拷锟诫径
	float circleDiameter = 80; //小圆锟斤拷锟斤拷
	float circleNarrow = 1.5; //圆锟诫径锟斤拷锟斤拷值
	int dispMin = 10; //锟接诧拷锟斤拷小值
	int dispMax = 25; //锟接诧拷锟斤拷锟街�
	float dispStep =1  ; //锟接诧拷锟斤拷锟街�            

	string folderName = "/home/jetson/Desktop/light";//锟斤拷锟斤拷屑锟斤拷锟斤拷锟斤拷募锟斤拷锟�
	string inputRawImg = "205.bmp";

	string centerPointFile = "points.txt";

	// 锟斤拷锟斤拷 dataParameter 锟窖讹拷锟藉并锟斤拷锟斤拷锟斤拷锟斤拷图锟斤拷 m_inputImg
	//cv::Mat m_inputImg = cv::imread("C:/work/LightFieldTool_CPU/LightFieldTool_CPU/205/205.bmp", cv::IMREAD_GRAYSCALE);
	/*
	if (m_inputImg.empty()) {
		std::cout << "Could not open or find the image" << std::endl;
		return ;
	}*/

	// 锟斤拷锟酵硷拷锟酵拷锟斤拷锟斤拷锟饺凤拷锟斤拷腔叶锟酵硷拷锟�
	/*
	if (m_inputImg.channels() == 1) {
		std::cout << "Input image is  a grayscale image" << std::endl;
	}*/

	


	//锟节讹拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟捷筹拷始锟斤拷
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟接差及锟斤拷锟脚讹拷mask锟斤拷锟姐，注锟斤拷锟叫伙拷锟疥定锟藉，注锟酵碉拷SCENE_DEPTH_COMPUTE锟侥讹拷锟藉即锟斤拷
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//锟斤拷锟侥诧拷锟斤拷锟斤拷锟斤拷Matlab锟斤拷锟斤拷锟斤拷锟斤拷涌拙锟酵硷拷锟斤拷锟斤拷染锟斤拷然锟斤拷锟接孔撅拷图锟斤拷锟接筹拷锟斤拷募锟斤拷锟斤拷诘锟角澳柯硷拷锟�
#else
	//锟斤拷锟藉步锟斤拷锟斤拷锟叫筹拷锟斤拷锟斤拷鹊募锟斤拷锟�,注锟斤拷锟叫伙拷前锟斤拷暮甓拷澹★拷锟阶拷锟絊CENE_DEPTH_COMPUTE锟斤拷锟斤拷

	string referSubImgName = "subAperatureImg.bmp";//锟接孔撅拷图锟斤拷
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //锟接诧拷锟侥硷拷
	std::string referMaskXmlName = "confidentMatMask.xml"; //锟斤拷锟脚讹拷锟侥硷拷
	string renderPointsMapping = "renderPointsMapping.txt";//锟斤拷染锟斤拷映锟斤拷锟�
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//锟斤拷锟斤拷锟斤拷 锟斤拷锟斤拷全锟斤拷锟脚伙拷Matlab锟斤拷锟斤拷
}

void ToolTwoTestDemo::data2compute()
{//锟斤拷锟斤拷1202
	//锟斤拷一锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟斤拷锟斤拷锟截诧拷锟斤拷锟斤拷锟斤拷锟斤拷
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x锟斤拷y锟斤拷锟斤拷锟较碉拷图锟斤拷偏锟斤拷值
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; //锟斤拷取锟斤拷图锟斤拷锟斤拷群透叨锟�

	int filterRadius = 8; //锟剿诧拷锟诫径 4
	float circleDiameter = 26.0; //小圆锟斤拷锟斤拷
	float circleNarrow = 1.5; //锟斤拷锟斤拷锟斤拷锟斤拷值
	int dispMin = 5; //锟接诧拷锟斤拷小值
	int dispMax = 8; //锟接诧拷锟斤拷锟街�
	float dispStep = 0.1; //锟接诧拷锟斤拷锟街�  0.1

	string folderName = "C:/work/LightFieldTool_CPU/LightFieldTool_CPU/725";//锟斤拷锟斤拷屑锟斤拷锟斤拷锟斤拷募锟斤拷锟�
	string inputRawImg = "725.bmp";

	string centerPointFile = "points.txt";


	//锟节讹拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟捷筹拷始锟斤拷
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟接差及锟斤拷锟脚讹拷mask锟斤拷锟姐，注锟斤拷锟叫伙拷锟疥定锟藉，注锟酵碉拷SCENE_DEPTH_COMPUTE锟侥讹拷锟藉即锟斤拷
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//锟斤拷锟侥诧拷锟斤拷锟斤拷锟斤拷Matlab锟斤拷锟斤拷锟斤拷锟斤拷涌拙锟酵硷拷锟斤拷锟斤拷染锟斤拷然锟斤拷锟接孔撅拷图锟斤拷锟接筹拷锟斤拷募锟斤拷锟斤拷诘锟角澳柯硷拷锟�
#else
	//锟斤拷锟藉步锟斤拷锟斤拷锟叫筹拷锟斤拷锟斤拷鹊募锟斤拷锟�,注锟斤拷锟叫伙拷前锟斤拷暮甓拷澹★拷锟阶拷锟絊CENE_DEPTH_COMPUTE锟斤拷锟斤拷

	string referSubImgName = "subAperatureImg.bmp";//锟接孔撅拷图锟斤拷
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //锟接诧拷锟侥硷拷
	std::string referMaskXmlName = "confidentMatMask.xml"; //锟斤拷锟脚讹拷锟侥硷拷
	string renderPointsMapping = "renderPointsMapping.txt";//锟斤拷染锟斤拷映锟斤拷锟�
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//锟斤拷锟斤拷锟斤拷 锟斤拷锟斤拷全锟斤拷锟脚伙拷Matlab锟斤拷锟斤拷
}

void ToolTwoTestDemo::data3compute()
{//锟斤拷锟斤拷 0321_4
	//锟斤拷一锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟斤拷锟斤拷锟截诧拷锟斤拷锟斤拷锟斤拷锟斤拷
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x锟斤拷y锟斤拷锟斤拷锟较碉拷图锟斤拷偏锟斤拷值
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; 

	int filterRadius = 4; //锟剿诧拷锟诫径
	float circleDiameter = 35; //小圆锟斤拷锟斤拷
	float circleNarrow = 1; //锟斤拷锟斤拷锟斤拷锟斤拷值
	int dispMin = 10; //锟接诧拷锟斤拷小值
	int dispMax = 13; //锟接诧拷锟斤拷锟街�
	float dispStep = 0.1; //锟接诧拷锟斤拷锟街�

	string folderName = "205_11";//锟斤拷锟斤拷屑锟斤拷锟斤拷锟斤拷募锟斤拷锟�
	string inputRawImg = "5m.bmp";
	string centerPointFile = "points.txt";

	//锟节讹拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟捷筹拷始锟斤拷
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟接差及锟斤拷锟脚讹拷mask锟斤拷锟姐，注锟斤拷锟叫伙拷锟疥定锟藉，注锟酵碉拷SCENE_DEPTH_COMPUTE锟侥讹拷锟藉即锟斤拷
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//锟斤拷锟侥诧拷锟斤拷锟斤拷锟斤拷Matlab锟斤拷锟斤拷锟斤拷锟斤拷涌拙锟酵硷拷锟斤拷锟斤拷染锟斤拷然锟斤拷锟接孔撅拷图锟斤拷锟接筹拷锟斤拷募锟斤拷锟斤拷诘锟角澳柯硷拷锟�
#else
	//锟斤拷锟藉步锟斤拷锟斤拷锟叫筹拷锟斤拷锟斤拷鹊募锟斤拷锟�,注锟斤拷锟叫伙拷前锟斤拷暮甓拷澹★拷锟阶拷锟絊CENE_DEPTH_COMPUTE锟斤拷锟斤拷

	string referSubImgName = "subAperatureImg.bmp";//锟接孔撅拷图锟斤拷
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //锟接诧拷锟侥硷拷
	std::string referMaskXmlName = "confidentMatMask.xml"; //锟斤拷锟脚讹拷锟侥硷拷
	string renderPointsMapping = "renderPointsMapping.txt";//锟斤拷染锟斤拷映锟斤拷锟�
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//锟斤拷锟斤拷锟斤拷 锟斤拷锟斤拷全锟斤拷锟脚伙拷Matlab锟斤拷锟斤拷
}

void ToolTwoTestDemo::data4compute()
{//锟斤拷锟斤拷 0321_5
	//锟斤拷一锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟斤拷锟斤拷锟截诧拷锟斤拷锟斤拷锟斤拷锟斤拷
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x锟斤拷y锟斤拷锟斤拷锟较碉拷图锟斤拷偏锟斤拷值
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; 

	int filterRadius = 8; //锟剿诧拷锟诫径
	float circleDiameter = 38; //小圆锟斤拷锟斤拷
	float circleNarrow = 1; //锟斤拷锟斤拷锟斤拷锟斤拷值
	int dispMin = 6; //锟接诧拷锟斤拷小值
	int dispMax = 10; //锟接诧拷锟斤拷锟街�
	float dispStep = 0.2; //锟接诧拷锟斤拷锟街�

	string folderName = "205_11_cut";//锟斤拷锟斤拷屑锟斤拷锟斤拷锟斤拷募锟斤拷锟�
	string inputRawImg = "1m_cut.bmp";
	string centerPointFile = "points.txt";

	//锟节讹拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟捷筹拷始锟斤拷
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟接差及锟斤拷锟脚讹拷mask锟斤拷锟姐，注锟斤拷锟叫伙拷锟疥定锟藉，注锟酵碉拷SCENE_DEPTH_COMPUTE锟侥讹拷锟藉即锟斤拷
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//锟斤拷锟侥诧拷锟斤拷锟斤拷锟斤拷Matlab锟斤拷锟斤拷锟斤拷锟斤拷涌拙锟酵硷拷锟斤拷锟斤拷染锟斤拷然锟斤拷锟接孔撅拷图锟斤拷锟接筹拷锟斤拷募锟斤拷锟斤拷诘锟角澳柯硷拷锟�
#else
	//锟斤拷锟藉步锟斤拷锟斤拷锟叫筹拷锟斤拷锟斤拷鹊募锟斤拷锟�,注锟斤拷锟叫伙拷前锟斤拷暮甓拷澹★拷锟阶拷锟絊CENE_DEPTH_COMPUTE锟斤拷锟斤拷

	string referSubImgName = "subAperatureImg.bmp";//锟接孔撅拷图锟斤拷
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //锟接诧拷锟侥硷拷
	std::string referMaskXmlName = "confidentMatMask.xml"; //锟斤拷锟脚讹拷锟侥硷拷
	string renderPointsMapping = "renderPointsMapping.txt";//锟斤拷染锟斤拷映锟斤拷锟�
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//锟斤拷锟斤拷锟斤拷 锟斤拷锟斤拷全锟斤拷锟脚伙拷Matlab锟斤拷锟斤拷
}

void ToolTwoTestDemo::data5compute()//主要函数
{//锟斤拷锟斤拷 0321_6
	//锟斤拷一锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟斤拷锟斤拷锟截诧拷锟斤拷锟斤拷锟斤拷锟斤拷
	int xCenterStartOffset = 2;
	int yCenterStartOffset = 2; //x锟斤拷y锟斤拷锟斤拷锟较碉拷图锟斤拷偏锟斤拷值
	int xCenterEndOffset = 2;
	int yCenterEndOffset = 2; //锟斤拷取锟斤拷图锟斤拷锟斤拷群透叨锟�


	/**正常参数 **/
	/*
	int filterRadius = 6; //锟剿诧拷锟诫径
	float circleDiameter = 40.0; //小圆锟斤拷锟斤拷
	float circleNarrow = 1.5; //锟斤拷锟斤拷锟斤拷锟斤拷值
	int dispMin = 5; //锟接诧拷锟斤拷小值
	int dispMax = 14; //锟接诧拷锟斤拷锟街�
	float dispStep = 0.25; //锟接诧拷锟斤拷锟街�
	
	*/
	/*简化版参数*/
	
	int filterRadius = 6; //滤波半径
	float circleDiameter = 2.0; //小圆宽度
	float circleNarrow = 1.0; //窗口缩减值
	int dispMin = 12; //视差最小值
	int dispMax = 13; //视差最大值
	float dispStep = 0.5; //视差迭代值
	
	

	string folderName = "/home/jetson/Desktop/light/205_66";//锟斤拷锟斤拷屑锟斤拷锟斤拷锟斤拷募锟斤拷锟�
	string inputRawImg = "66_5.bmp";
	string centerPointFile = "points.txt";

	//锟节讹拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟捷筹拷始锟斤拷
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟接差及锟斤拷锟脚讹拷mask锟斤拷锟姐，注锟斤拷锟叫伙拷锟疥定锟藉，注锟酵碉拷SCENE_DEPTH_COMPUTE锟侥讹拷锟藉即锟斤拷
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//锟斤拷锟侥诧拷锟斤拷锟斤拷锟斤拷Matlab锟斤拷锟斤拷锟斤拷锟斤拷涌拙锟酵硷拷锟斤拷锟斤拷染锟斤拷然锟斤拷锟接孔撅拷图锟斤拷锟接筹拷锟斤拷募锟斤拷锟斤拷诘锟角澳柯硷拷锟�
#else
	//锟斤拷锟藉步锟斤拷锟斤拷锟叫筹拷锟斤拷锟斤拷鹊募锟斤拷锟�,注锟斤拷锟叫伙拷前锟斤拷暮甓拷澹★拷锟阶拷锟絊CENE_DEPTH_COMPUTE锟斤拷锟斤拷

	string referSubImgName = "subAperatureImg.bmp";//锟接孔撅拷图锟斤拷
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //锟接诧拷锟侥硷拷
	std::string referMaskXmlName = "confidentMatMask.xml"; //锟斤拷锟脚讹拷锟侥硷拷
	string renderPointsMapping = "renderPointsMapping.txt";//锟斤拷染锟斤拷映锟斤拷锟�
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//锟斤拷锟斤拷锟斤拷 锟斤拷锟斤拷全锟斤拷锟脚伙拷Matlab锟斤拷锟斤拷
}


void ToolTwoTestDemo::data6compute()
{
	//锟斤拷一锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟斤拷锟斤拷锟截诧拷锟斤拷锟斤拷锟斤拷锟斤拷
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x锟斤拷y锟斤拷锟斤拷锟较碉拷图锟斤拷偏锟斤拷值
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; //锟斤拷取锟斤拷图锟斤拷锟斤拷群透叨锟�

	int filterRadius = 8; //锟剿诧拷锟诫径
	float circleDiameter = 40.0; //小圆锟斤拷锟斤拷
	float circleNarrow = 1.5; //锟斤拷锟斤拷锟斤拷锟斤拷值
	int dispMin = 6; //锟接诧拷锟斤拷小值
	int dispMax = 10; //锟接诧拷锟斤拷锟街�
	float dispStep = 0.5; //锟接诧拷锟斤拷锟街�

	string folderName = "205_11_77";//锟斤拷锟斤拷屑锟斤拷锟斤拷锟斤拷募锟斤拷锟�
	string inputRawImg = "205.bmp";
	string centerPointFile = "points.txt";

	//锟节讹拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟捷筹拷始锟斤拷
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟接差及锟斤拷锟脚讹拷mask锟斤拷锟姐，注锟斤拷锟叫伙拷锟疥定锟藉，注锟酵碉拷SCENE_DEPTH_COMPUTE锟侥讹拷锟藉即锟斤拷
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//锟斤拷锟侥诧拷锟斤拷锟斤拷锟斤拷Matlab锟斤拷锟斤拷锟斤拷锟斤拷涌拙锟酵硷拷锟斤拷锟斤拷染锟斤拷然锟斤拷锟接孔撅拷图锟斤拷锟接筹拷锟斤拷募锟斤拷锟斤拷诘锟角澳柯硷拷锟�
#else
	//锟斤拷锟藉步锟斤拷锟斤拷锟叫筹拷锟斤拷锟斤拷鹊募锟斤拷锟�,注锟斤拷锟叫伙拷前锟斤拷暮甓拷澹★拷锟阶拷锟絊CENE_DEPTH_COMPUTE锟斤拷锟斤拷

	string referSubImgName = "subAperatureImg.bmp";//锟接孔撅拷图锟斤拷
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //锟接诧拷锟侥硷拷
	std::string referMaskXmlName = "confidentMatMask.xml"; //锟斤拷锟脚讹拷锟侥硷拷
	string renderPointsMapping = "renderPointsMapping.txt";//锟斤拷染锟斤拷映锟斤拷锟�
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//锟斤拷锟斤拷锟斤拷 锟斤拷锟斤拷全锟斤拷锟脚伙拷Matlab锟斤拷锟斤拷
}

void ToolTwoTestDemo::data7compute()
{//锟斤拷锟斤拷 0321_6
	//锟斤拷一锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟斤拷锟斤拷锟截诧拷锟斤拷锟斤拷锟斤拷锟斤拷
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x锟斤拷y锟斤拷锟斤拷锟较碉拷图锟斤拷偏锟斤拷值
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; //锟斤拷取锟斤拷图锟斤拷锟斤拷群透叨锟�

	int filterRadius = 8; //锟剿诧拷锟诫径
	float circleDiameter = 40.0; //小圆锟斤拷锟斤拷
	float circleNarrow = 1.5; //锟斤拷锟斤拷锟斤拷锟斤拷值 
	/*
	int dispMin = 7; //锟接诧拷锟斤拷小值
	int dispMax = 14; //锟接诧拷锟斤拷锟街�
	float dispStep = 0.5; //锟接诧拷锟斤拷锟街�
	*/
	int dispMin = 7; //锟接诧拷锟斤拷小值
	int dispMax = 8; //锟接诧拷锟斤拷锟街�
	float dispStep = 0.5; //锟接诧拷锟斤拷锟街�


	string folderName = "205_5m";//锟斤拷锟斤拷屑锟斤拷锟斤拷锟斤拷募锟斤拷锟�
	string inputRawImg = "205_5m.bmp";
	string centerPointFile = "points.txt";

	//锟节讹拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟捷筹拷始锟斤拷
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟接差及锟斤拷锟脚讹拷mask锟斤拷锟姐，注锟斤拷锟叫伙拷锟疥定锟藉，注锟酵碉拷SCENE_DEPTH_COMPUTE锟侥讹拷锟藉即锟斤拷
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//锟斤拷锟侥诧拷锟斤拷锟斤拷锟斤拷Matlab锟斤拷锟斤拷锟斤拷锟斤拷涌拙锟酵硷拷锟斤拷锟斤拷染锟斤拷然锟斤拷锟接孔撅拷图锟斤拷锟接筹拷锟斤拷募锟斤拷锟斤拷诘锟角澳柯硷拷锟�
#else
	//锟斤拷锟藉步锟斤拷锟斤拷锟叫筹拷锟斤拷锟斤拷鹊募锟斤拷锟�,注锟斤拷锟叫伙拷前锟斤拷暮甓拷澹★拷锟阶拷锟絊CENE_DEPTH_COMPUTE锟斤拷锟斤拷

	string referSubImgName = "subAperatureImg.bmp";//锟接孔撅拷图锟斤拷
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //锟接诧拷锟侥硷拷
	std::string referMaskXmlName = "confidentMatMask.xml"; //锟斤拷锟脚讹拷锟侥硷拷
	string renderPointsMapping = "renderPointsMapping.txt";//锟斤拷染锟斤拷映锟斤拷锟�
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//锟斤拷锟斤拷锟斤拷 锟斤拷锟斤拷全锟斤拷锟脚伙拷Matlab锟斤拷锟斤拷
}


void ToolTwoTestDemo::data8compute()
{//
	//锟斤拷一锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟斤拷锟斤拷锟截诧拷锟斤拷锟斤拷锟斤拷锟斤拷
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x锟斤拷y锟斤拷锟斤拷锟较碉拷图锟斤拷偏锟斤拷值
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; //锟斤拷取锟斤拷图锟斤拷锟斤拷群透叨锟�

	int filterRadius = 8; //锟剿诧拷锟诫径
	float circleDiameter = 40.0; //小圆锟斤拷锟斤拷
	float circleNarrow = 1.5; //锟斤拷锟斤拷锟斤拷锟斤拷值
	int dispMin = 8; //锟接诧拷锟斤拷小值
	int dispMax = 14; //锟接诧拷锟斤拷锟街�
	float dispStep = 0.5; //锟接诧拷锟斤拷锟街�

	string folderName = "205_15m";//锟斤拷锟斤拷屑锟斤拷锟斤拷锟斤拷募锟斤拷锟�
	string inputRawImg = "15.bmp";
	string centerPointFile = "points.txt";

	//锟节讹拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟捷筹拷始锟斤拷
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟接差及锟斤拷锟脚讹拷mask锟斤拷锟姐，注锟斤拷锟叫伙拷锟疥定锟藉，注锟酵碉拷SCENE_DEPTH_COMPUTE锟侥讹拷锟藉即锟斤拷
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//锟斤拷锟侥诧拷锟斤拷锟斤拷锟斤拷Matlab锟斤拷锟斤拷锟斤拷锟斤拷涌拙锟酵硷拷锟斤拷锟斤拷染锟斤拷然锟斤拷锟接孔撅拷图锟斤拷锟接筹拷锟斤拷募锟斤拷锟斤拷诘锟角澳柯硷拷锟�
#else
	//锟斤拷锟藉步锟斤拷锟斤拷锟叫筹拷锟斤拷锟斤拷鹊募锟斤拷锟�,注锟斤拷锟叫伙拷前锟斤拷暮甓拷澹★拷锟阶拷锟絊CENE_DEPTH_COMPUTE锟斤拷锟斤拷

	string referSubImgName = "subAperatureImg.bmp";//锟接孔撅拷图锟斤拷
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //锟接诧拷锟侥硷拷
	std::string referMaskXmlName = "confidentMatMask.xml"; //锟斤拷锟脚讹拷锟侥硷拷
	string renderPointsMapping = "renderPointsMapping.txt";//锟斤拷染锟斤拷映锟斤拷锟�
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//锟斤拷锟斤拷锟斤拷 锟斤拷锟斤拷全锟斤拷锟脚伙拷Matlab锟斤拷锟斤拷
}


void ToolTwoTestDemo::data9compute()
{//
	//锟斤拷一锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟斤拷锟斤拷锟截诧拷锟斤拷锟斤拷锟斤拷锟斤拷
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x锟斤拷y锟斤拷锟斤拷锟较碉拷图锟斤拷偏锟斤拷值
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; //锟斤拷取锟斤拷图锟斤拷锟斤拷群透叨锟�

	int filterRadius = 8; //锟剿诧拷锟诫径
	float circleDiameter = 40.0; //小圆锟斤拷锟斤拷
	float circleNarrow = 1.5; //锟斤拷锟斤拷锟斤拷锟斤拷值
	int dispMin = 7; //锟接诧拷锟斤拷小值
	int dispMax = 15; //锟接诧拷锟斤拷锟街�
	float dispStep = 0.5; //锟接诧拷锟斤拷锟街�

	string folderName = "205_25m";//锟斤拷锟斤拷屑锟斤拷锟斤拷锟斤拷募锟斤拷锟�
	string inputRawImg = "25.bmp";
	string centerPointFile = "points.txt";

	//锟节讹拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟捷筹拷始锟斤拷
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟接差及锟斤拷锟脚讹拷mask锟斤拷锟姐，注锟斤拷锟叫伙拷锟疥定锟藉，注锟酵碉拷SCENE_DEPTH_COMPUTE锟侥讹拷锟藉即锟斤拷
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//锟斤拷锟侥诧拷锟斤拷锟斤拷锟斤拷Matlab锟斤拷锟斤拷锟斤拷锟斤拷涌拙锟酵硷拷锟斤拷锟斤拷染锟斤拷然锟斤拷锟接孔撅拷图锟斤拷锟接筹拷锟斤拷募锟斤拷锟斤拷诘锟角澳柯硷拷锟�
#else
	//锟斤拷锟藉步锟斤拷锟斤拷锟叫筹拷锟斤拷锟斤拷鹊募锟斤拷锟�,注锟斤拷锟叫伙拷前锟斤拷暮甓拷澹★拷锟阶拷锟絊CENE_DEPTH_COMPUTE锟斤拷锟斤拷

	string referSubImgName = "subAperatureImg.bmp";//锟接孔撅拷图锟斤拷
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //锟接诧拷锟侥硷拷
	std::string referMaskXmlName = "confidentMatMask.xml"; //锟斤拷锟脚讹拷锟侥硷拷
	string renderPointsMapping = "renderPointsMapping.txt";//锟斤拷染锟斤拷映锟斤拷锟�
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//锟斤拷锟斤拷锟斤拷 锟斤拷锟斤拷全锟斤拷锟脚伙拷Matlab锟斤拷锟斤拷
}

void ToolTwoTestDemo::data10compute()
{//
	//锟斤拷一锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟斤拷锟斤拷锟截诧拷锟斤拷锟斤拷锟斤拷锟斤拷
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x锟斤拷y锟斤拷锟斤拷锟较碉拷图锟斤拷偏锟斤拷值
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; //锟斤拷取锟斤拷图锟斤拷锟斤拷群透叨锟�

	int filterRadius = 8; //锟剿诧拷锟诫径
	float circleDiameter = 40.0; //小圆锟斤拷锟斤拷
	float circleNarrow = 1.5; //锟斤拷锟斤拷锟斤拷锟斤拷值
	int dispMin = 7; //锟接诧拷锟斤拷小值
	int dispMax = 15; //锟接诧拷锟斤拷锟街�
	float dispStep = 0.5; //锟接诧拷锟斤拷锟街�

	string folderName = "205_50m";//锟斤拷锟斤拷屑锟斤拷锟斤拷锟斤拷募锟斤拷锟�
	string inputRawImg = "50.bmp";
	string centerPointFile = "points.txt";

	//锟节讹拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟捷筹拷始锟斤拷
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷Raw图锟斤拷锟接差及锟斤拷锟脚讹拷mask锟斤拷锟姐，注锟斤拷锟叫伙拷锟疥定锟藉，注锟酵碉拷SCENE_DEPTH_COMPUTE锟侥讹拷锟藉即锟斤拷
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//锟斤拷锟侥诧拷锟斤拷锟斤拷锟斤拷Matlab锟斤拷锟斤拷锟斤拷锟斤拷涌拙锟酵硷拷锟斤拷锟斤拷染锟斤拷然锟斤拷锟接孔撅拷图锟斤拷锟接筹拷锟斤拷募锟斤拷锟斤拷诘锟角澳柯硷拷锟�
#else
	//锟斤拷锟藉步锟斤拷锟斤拷锟叫筹拷锟斤拷锟斤拷鹊募锟斤拷锟�,注锟斤拷锟叫伙拷前锟斤拷暮甓拷澹★拷锟阶拷锟絊CENE_DEPTH_COMPUTE锟斤拷锟斤拷

	string referSubImgName = "subAperatureImg.bmp";//锟接孔撅拷图锟斤拷
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //锟接诧拷锟侥硷拷
	std::string referMaskXmlName = "confidentMatMask.xml"; //锟斤拷锟脚讹拷锟侥硷拷
	string renderPointsMapping = "renderPointsMapping.txt";//锟斤拷染锟斤拷映锟斤拷锟�
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//锟斤拷锟斤拷锟斤拷 锟斤拷锟斤拷全锟斤拷锟脚伙拷Matlab锟斤拷锟斤拷
}