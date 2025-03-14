#include "ToolOneTestDemo.h"
#include "DepthComputeToolOne.h"

using namespace std;
using namespace cv;

ToolOneTestDemo::ToolOneTestDemo()
{
}

ToolOneTestDemo::~ToolOneTestDemo()
{
}

 // #define SCENE_DEPTH_COMPUTE 1

void ToolOneTestDemo::data1compute()
{//数据1106
	//第一步，对Raw图像进行相关参数的设置
	int xCenterStartOffset = 3;
	int yCenterStartOffset = 3; //x和y方向上的图像偏移值
	int xCenterEndOffset = 3;
	int yCenterEndOffset = 3; //截取的图像宽度和高度

	int filterRadius = 4; //滤波半径
	float circleDiameter = 34.0; //小圆宽度
	float circleNarrow = 1.5; //窗口缩减值
	int dispMin = 5; //视差最小值
	int dispMax = 13; //视差最大值
	float dispStep = 0.5; //视差迭代值

	string folderName = "data1106";//存放中间结果的文件夹
	string inputRawImg = "src_1106_rct.bmp";
	string centerPointFile = "white_1106_rct_center.txt";
	
	//第二步，进行数据初始化
	DepthComputeToolOne depthComputeToolOne;
	depthComputeToolOne.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//第三步，进行Raw图像的视差计算，注意切换宏定义，注释掉SCENE_DEPTH_COMPUTE的定义即可
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolOne.rawImageDisparityCompute();

	//第四步，采用Matlab程序进行子孔径图像的渲染，然后将子孔径图像和映射文件放在当前目录下
#else
	//第五步，进行场景深度的计算,注意切换前面的宏定义，取消注释SCENE_DEPTH_COMPUTE即可
	string subImageName = "randerSubImg.bmp";//子孔径图像
	string renderPointsMapping = "renderPointsMapping.txt";//渲染的映射点
	depthComputeToolOne.sceneDepthCompute(subImageName, renderPointsMapping);
#endif
}

void ToolOneTestDemo::data2compute()
{//数据0822
	//第一步，对Raw图像进行相关参数的设置
	int xCenterStartOffset = 3;
	int yCenterStartOffset = 3; //x和y方向上的图像偏移值
	int xCenterEndOffset = 3;
	int yCenterEndOffset = 3; //截取的图像宽度和高度

	int filterRadius = 4; //滤波半径
	float circleDiameter = 34.0; //小圆宽度
	float circleNarrow = 1.5; //窗口缩减值
	int dispMin = 5; //视差最小值
	int dispMax = 13; //视差最大值
	float dispStep = 0.5; //视差迭代值

	string folderName = "data0822";//存放中间结果的文件夹
	string inputRawImg = "src_0822_rct.bmp";
	string centerPointFile = "white_0822_rct_center.txt";

	//第二步，进行数据初始化
	DepthComputeToolOne depthComputeToolOne;
	depthComputeToolOne.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//第三步，进行Raw图像的视差计算，注意切换宏定义，注释掉SCENE_DEPTH_COMPUTE的定义即可
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolOne.rawImageDisparityCompute();

	//第四步，采用Matlab程序进行子孔径图像的渲染，然后将子孔径图像和映射文件放在当前目录下
#else
	//第五步，进行场景深度的计算,注意切换前面的宏定义，取消注释SCENE_DEPTH_COMPUTE即可
	string subImageName = "randerSubImg.bmp";//子孔径图像
	string renderPointsMapping = "renderPointsMapping.txt";//渲染的映射点
	depthComputeToolOne.sceneDepthCompute(subImageName, renderPointsMapping);
#endif
}


void ToolOneTestDemo::data3compute()
{//数据1130_1
	//第一步，对Raw图像进行相关参数的设置
	int xCenterStartOffset = 3;
	int yCenterStartOffset = 3; //x和y方向上的图像偏移值
	int xCenterEndOffset = 3;
	int yCenterEndOffset = 3; //截取的图像宽度和高度

	int filterRadius = 4; //滤波半径
	float circleDiameter = 34.0; //小圆宽度
	float circleNarrow = 1.5; //窗口缩减值
	int dispMin = 5; //视差最小值
	int dispMax = 14; //视差最大值
	float dispStep = 0.5; //视差迭代值

	string folderName = "data1130_1";//存放中间结果的文件夹
	string inputRawImg = "1130_1_src_rct.bmp";
	string centerPointFile = "1130_1_white_rct_center.txt";

	//第二步，进行数据初始化
	DepthComputeToolOne depthComputeToolOne;
	depthComputeToolOne.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//第三步，进行Raw图像的视差计算，注意切换宏定义，注释掉SCENE_DEPTH_COMPUTE的定义即可
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolOne.rawImageDisparityCompute();

	//第四步，采用Matlab程序进行子孔径图像的渲染，然后将子孔径图像和映射文件放在当前目录下
#else
	//第五步，进行场景深度的计算,注意切换前面的宏定义，取消注释SCENE_DEPTH_COMPUTE即可
	string subImageName = "randerSubImg.bmp";//子孔径图像
	string renderPointsMapping = "renderPointsMapping.txt";//渲染的映射点
	depthComputeToolOne.sceneDepthCompute(subImageName, renderPointsMapping);
#endif
}

void ToolOneTestDemo::data4compute()
{//数据1130_2
	//第一步，对Raw图像进行相关参数的设置
	int xCenterStartOffset = 3;
	int yCenterStartOffset = 3; //x和y方向上的图像偏移值
	int xCenterEndOffset = 3;
	int yCenterEndOffset = 3; //截取的图像宽度和高度

	int filterRadius = 4; //滤波半径
	float circleDiameter = 34.0; //小圆宽度
	float circleNarrow = 1.5; //窗口缩减值
	int dispMin = 5; //视差最小值
	int dispMax = 14; //视差最大值
	float dispStep = 0.5; //视差迭代值

	string folderName = "data1130_2";//存放中间结果的文件夹
	string inputRawImg = "1130_2_src_rct.bmp";
	string centerPointFile = "1130_2_white_rct_center.txt";

	//第二步，进行数据初始化
	DepthComputeToolOne depthComputeToolOne;
	depthComputeToolOne.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//第三步，进行Raw图像的视差计算，注意切换宏定义，注释掉SCENE_DEPTH_COMPUTE的定义即可
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolOne.rawImageDisparityCompute();

	//第四步，采用Matlab程序进行子孔径图像的渲染，然后将子孔径图像和映射文件放在当前目录下
#else
	//第五步，进行场景深度的计算,注意切换前面的宏定义，取消注释SCENE_DEPTH_COMPUTE即可
	string subImageName = "randerSubImg.bmp";//子孔径图像
	string renderPointsMapping = "renderPointsMapping.txt";//渲染的映射点
	depthComputeToolOne.sceneDepthCompute(subImageName, renderPointsMapping);
#endif
}

void ToolOneTestDemo::data5compute()
{//数据1130_3
	//第一步，对Raw图像进行相关参数的设置
	int xCenterStartOffset = 3;
	int yCenterStartOffset = 3; //x和y方向上的图像偏移值
	int xCenterEndOffset = 3;
	int yCenterEndOffset = 3; //截取的图像宽度和高度

	int filterRadius = 4; //滤波半径
	float circleDiameter = 34.0; //小圆宽度
	float circleNarrow = 1.5; //窗口缩减值
	int dispMin = 5; //视差最小值
	int dispMax = 14; //视差最大值
	float dispStep = 0.5; //视差迭代值

	string folderName = "data1130_3";//存放中间结果的文件夹
	string inputRawImg = "1130_3_src_rct.bmp";
	string centerPointFile = "1130_3_white_rct_center.txt";

	//第二步，进行数据初始化
	DepthComputeToolOne depthComputeToolOne;
	depthComputeToolOne.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//第三步，进行Raw图像的视差计算，注意切换宏定义，注释掉SCENE_DEPTH_COMPUTE的定义即可
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolOne.rawImageDisparityCompute();

	//第四步，采用Matlab程序进行子孔径图像的渲染，然后将子孔径图像和映射文件放在当前目录下
#else
	//第五步，进行场景深度的计算,注意切换前面的宏定义，取消注释SCENE_DEPTH_COMPUTE即可
	string subImageName = "randerSubImg.bmp";//子孔径图像
	string renderPointsMapping = "renderPointsMapping.txt";//渲染的映射点
	depthComputeToolOne.sceneDepthCompute(subImageName, renderPointsMapping);
#endif
}

void ToolOneTestDemo::data6compute()
{//数据1130_4
	//第一步，对Raw图像进行相关参数的设置
	int xCenterStartOffset = 3;
	int yCenterStartOffset = 3; //x和y方向上的图像偏移值
	int xCenterEndOffset = 3;
	int yCenterEndOffset = 3; //截取的图像宽度和高度

	int filterRadius = 4; //滤波半径
	float circleDiameter = 34.0; //小圆宽度
	float circleNarrow = 1.5; //窗口缩减值
	int dispMin = 5; //视差最小值
	int dispMax = 14; //视差最大值
	float dispStep = 0.5; //视差迭代值

	string folderName = "data1130_4";//存放中间结果的文件夹
	string inputRawImg = "1130_4_src_rct.bmp";
	string centerPointFile = "1130_4_white_rct_center.txt";

	//第二步，进行数据初始化
	DepthComputeToolOne depthComputeToolOne;
	depthComputeToolOne.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//第三步，进行Raw图像的视差计算，注意切换宏定义，注释掉SCENE_DEPTH_COMPUTE的定义即可
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolOne.rawImageDisparityCompute();

	//第四步，采用Matlab程序进行子孔径图像的渲染，然后将子孔径图像和映射文件放在当前目录下
#else
	//第五步，进行场景深度的计算,注意切换前面的宏定义，取消注释SCENE_DEPTH_COMPUTE即可
	string subImageName = "randerSubImg.bmp";//子孔径图像
	string renderPointsMapping = "renderPointsMapping.txt";//渲染的映射点
	depthComputeToolOne.sceneDepthCompute(subImageName, renderPointsMapping);
#endif
}

void ToolOneTestDemo::data7compute()
{//数据1130_6
	//第一步，对Raw图像进行相关参数的设置
	int xCenterStartOffset = 3;
	int yCenterStartOffset = 3; //x和y方向上的图像偏移值
	int xCenterEndOffset = 3;
	int yCenterEndOffset = 3; //截取的图像宽度和高度

	int filterRadius = 4; //滤波半径
	float circleDiameter = 34.0; //小圆宽度
	float circleNarrow = 1.5; //窗口缩减值
	int dispMin = 5; //视差最小值
	int dispMax = 14; //视差最大值
	float dispStep = 0.5; //视差迭代值

	string folderName = "data1130_6";//存放中间结果的文件夹
	string inputRawImg = "1130_6_src_rct.bmp";
	string centerPointFile = "1130_6_white_rct_center.txt";

	//第二步，进行数据初始化
	DepthComputeToolOne depthComputeToolOne;
	depthComputeToolOne.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//第三步，进行Raw图像的视差计算，注意切换宏定义，注释掉SCENE_DEPTH_COMPUTE的定义即可
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolOne.rawImageDisparityCompute();

	//第四步，采用Matlab程序进行子孔径图像的渲染，然后将子孔径图像和映射文件放在当前目录下
#else
	//第五步，进行场景深度的计算,注意切换前面的宏定义，取消注释SCENE_DEPTH_COMPUTE即可
	string subImageName = "randerSubImg.bmp";//子孔径图像
	string renderPointsMapping = "renderPointsMapping.txt";//渲染的映射点
	depthComputeToolOne.sceneDepthCompute(subImageName, renderPointsMapping);
#endif
}

void ToolOneTestDemo::data8compute()
{//数据1130_7
	//第一步，对Raw图像进行相关参数的设置
	int xCenterStartOffset = 3;
	int yCenterStartOffset = 3; //x和y方向上的图像偏移值
	int xCenterEndOffset = 3;
	int yCenterEndOffset = 3; //截取的图像宽度和高度

	int filterRadius = 4; //滤波半径
	float circleDiameter = 34.0; //小圆宽度
	float circleNarrow = 1.5; //窗口缩减值
	int dispMin = 5; //视差最小值
	int dispMax = 14; //视差最大值
	float dispStep = 0.5; //视差迭代值

	string folderName = "data1130_7";//存放中间结果的文件夹
	string inputRawImg = "1130_7_src_rct.bmp";
	string centerPointFile = "1130_7_white_rct_center.txt";

	//第二步，进行数据初始化
	DepthComputeToolOne depthComputeToolOne;
	depthComputeToolOne.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//第三步，进行Raw图像的视差计算，注意切换宏定义，注释掉SCENE_DEPTH_COMPUTE的定义即可
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolOne.rawImageDisparityCompute();

	//第四步，采用Matlab程序进行子孔径图像的渲染，然后将子孔径图像和映射文件放在当前目录下
#else
	//第五步，进行场景深度的计算,注意切换前面的宏定义，取消注释SCENE_DEPTH_COMPUTE即可
	string subImageName = "randerSubImg.bmp";//子孔径图像
	string renderPointsMapping = "renderPointsMapping.txt";//渲染的映射点
	depthComputeToolOne.sceneDepthCompute(subImageName, renderPointsMapping);
#endif
}