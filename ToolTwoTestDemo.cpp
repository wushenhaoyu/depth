// Software Version: V1.00
// SoftWare Name:������ʾ����TS-RJ4001
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
{//����0321_2
	//��һ������Rawͼ�������ز���������
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x��y�����ϵ�ͼ��ƫ��ֵ
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; //��ȡ��ͼ����Ⱥ͸߶�

	int filterRadius = 6; //�˲��뾶
	float circleDiameter = 80; //СԲ����
	float circleNarrow = 1.5; //Բ�뾶����ֵ
	int dispMin = 10; //�Ӳ���Сֵ
	int dispMax = 25; //�Ӳ����ֵ
	float dispStep =1  ; //�Ӳ����ֵ            

	string folderName = "/home/jetson/Desktop/depth";//����м������ļ���
	string inputRawImg = "205.bmp";

	string centerPointFile = "points.txt";

	// ���� dataParameter �Ѷ��岢��������ͼ�� m_inputImg
	//cv::Mat m_inputImg = cv::imread("C:/work/LightFieldTool_CPU/LightFieldTool_CPU/205/205.bmp", cv::IMREAD_GRAYSCALE);
	/*
	if (m_inputImg.empty()) {
		std::cout << "Could not open or find the image" << std::endl;
		return ;
	}*/

	// ���ͼ��ͨ������ȷ���ǻҶ�ͼ��
	/*
	if (m_inputImg.channels() == 1) {
		std::cout << "Input image is  a grayscale image" << std::endl;
	}*/

	


	//�ڶ������������ݳ�ʼ��
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//������������Rawͼ���Ӳ���Ŷ�mask���㣬ע���л��궨�壬ע�͵�SCENE_DEPTH_COMPUTE�Ķ��弴��
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//���Ĳ�������Matlab��������ӿ׾�ͼ�����Ⱦ��Ȼ���ӿ׾�ͼ���ӳ���ļ����ڵ�ǰĿ¼��
#else
	//���岽�����г�����ȵļ���,ע���л�ǰ��ĺ궨�壬ȡ��ע��SCENE_DEPTH_COMPUTE����

	string referSubImgName = "subAperatureImg.bmp";//�ӿ׾�ͼ��
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //�Ӳ��ļ�
	std::string referMaskXmlName = "confidentMatMask.xml"; //���Ŷ��ļ�
	string renderPointsMapping = "renderPointsMapping.txt";//��Ⱦ��ӳ���
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//������ ����ȫ���Ż�Matlab����
}

void ToolTwoTestDemo::data2compute()
{//����1202
	//��һ������Rawͼ�������ز���������
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x��y�����ϵ�ͼ��ƫ��ֵ
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; //��ȡ��ͼ����Ⱥ͸߶�

	int filterRadius = 8; //�˲��뾶 4
	float circleDiameter = 26.0; //СԲ����
	float circleNarrow = 1.5; //��������ֵ
	int dispMin = 5; //�Ӳ���Сֵ
	int dispMax = 8; //�Ӳ����ֵ
	float dispStep = 0.1; //�Ӳ����ֵ  0.1

	string folderName = "C:/work/LightFieldTool_CPU/LightFieldTool_CPU/725";//����м������ļ���
	string inputRawImg = "725.bmp";

	string centerPointFile = "points.txt";


	//�ڶ������������ݳ�ʼ��
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//������������Rawͼ���Ӳ���Ŷ�mask���㣬ע���л��궨�壬ע�͵�SCENE_DEPTH_COMPUTE�Ķ��弴��
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//���Ĳ�������Matlab��������ӿ׾�ͼ�����Ⱦ��Ȼ���ӿ׾�ͼ���ӳ���ļ����ڵ�ǰĿ¼��
#else
	//���岽�����г�����ȵļ���,ע���л�ǰ��ĺ궨�壬ȡ��ע��SCENE_DEPTH_COMPUTE����

	string referSubImgName = "subAperatureImg.bmp";//�ӿ׾�ͼ��
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //�Ӳ��ļ�
	std::string referMaskXmlName = "confidentMatMask.xml"; //���Ŷ��ļ�
	string renderPointsMapping = "renderPointsMapping.txt";//��Ⱦ��ӳ���
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//������ ����ȫ���Ż�Matlab����
}

void ToolTwoTestDemo::data3compute()
{//���� 0321_4
	//��һ������Rawͼ�������ز���������
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x��y�����ϵ�ͼ��ƫ��ֵ
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; 

	int filterRadius = 4; //�˲��뾶
	float circleDiameter = 35; //СԲ����
	float circleNarrow = 1; //��������ֵ
	int dispMin = 10; //�Ӳ���Сֵ
	int dispMax = 13; //�Ӳ����ֵ
	float dispStep = 0.1; //�Ӳ����ֵ

	string folderName = "205_11";//����м������ļ���
	string inputRawImg = "5m.bmp";
	string centerPointFile = "points.txt";

	//�ڶ������������ݳ�ʼ��
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//������������Rawͼ���Ӳ���Ŷ�mask���㣬ע���л��궨�壬ע�͵�SCENE_DEPTH_COMPUTE�Ķ��弴��
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//���Ĳ�������Matlab��������ӿ׾�ͼ�����Ⱦ��Ȼ���ӿ׾�ͼ���ӳ���ļ����ڵ�ǰĿ¼��
#else
	//���岽�����г�����ȵļ���,ע���л�ǰ��ĺ궨�壬ȡ��ע��SCENE_DEPTH_COMPUTE����

	string referSubImgName = "subAperatureImg.bmp";//�ӿ׾�ͼ��
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //�Ӳ��ļ�
	std::string referMaskXmlName = "confidentMatMask.xml"; //���Ŷ��ļ�
	string renderPointsMapping = "renderPointsMapping.txt";//��Ⱦ��ӳ���
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//������ ����ȫ���Ż�Matlab����
}

void ToolTwoTestDemo::data4compute()
{//���� 0321_5
	//��һ������Rawͼ�������ز���������
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x��y�����ϵ�ͼ��ƫ��ֵ
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; 

	int filterRadius = 8; //�˲��뾶
	float circleDiameter = 38; //СԲ����
	float circleNarrow = 1; //��������ֵ
	int dispMin = 6; //�Ӳ���Сֵ
	int dispMax = 10; //�Ӳ����ֵ
	float dispStep = 0.2; //�Ӳ����ֵ

	string folderName = "205_11_cut";//����м������ļ���
	string inputRawImg = "1m_cut.bmp";
	string centerPointFile = "points.txt";

	//�ڶ������������ݳ�ʼ��
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//������������Rawͼ���Ӳ���Ŷ�mask���㣬ע���л��궨�壬ע�͵�SCENE_DEPTH_COMPUTE�Ķ��弴��
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//���Ĳ�������Matlab��������ӿ׾�ͼ�����Ⱦ��Ȼ���ӿ׾�ͼ���ӳ���ļ����ڵ�ǰĿ¼��
#else
	//���岽�����г�����ȵļ���,ע���л�ǰ��ĺ궨�壬ȡ��ע��SCENE_DEPTH_COMPUTE����

	string referSubImgName = "subAperatureImg.bmp";//�ӿ׾�ͼ��
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //�Ӳ��ļ�
	std::string referMaskXmlName = "confidentMatMask.xml"; //���Ŷ��ļ�
	string renderPointsMapping = "renderPointsMapping.txt";//��Ⱦ��ӳ���
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//������ ����ȫ���Ż�Matlab����
}

void ToolTwoTestDemo::data5compute()
{//���� 0321_6
	//��һ������Rawͼ�������ز���������
	int xCenterStartOffset = 2;
	int yCenterStartOffset = 2; //x��y�����ϵ�ͼ��ƫ��ֵ
	int xCenterEndOffset = 2;
	int yCenterEndOffset = 2; //��ȡ��ͼ����Ⱥ͸߶�


	/**�������� **/
	
	int filterRadius = 6; //�˲��뾶
	float circleDiameter = 40.0; //СԲ����
	float circleNarrow = 1.5; //��������ֵ
	int dispMin = 5; //�Ӳ���Сֵ
	int dispMax = 14; //�Ӳ����ֵ
	float dispStep = 0.25; //�Ӳ����ֵ
	

	/*ֻ����һ���ؾ۽�ͼ�Ĳ���*/
	/*
	int filterRadius = 2; //�˲��뾶
	float circleDiameter = 2.0; //СԲ����
	float circleNarrow = 1.0; //��������ֵ
	int dispMin = 12; //�Ӳ���Сֵ
	int dispMax = 13; //�Ӳ����ֵ
	float dispStep = 0.5; //�Ӳ����ֵ
	*/
	

	string folderName = "/home/jetson/Desktop/depth/205_66";//����м������ļ���
	string inputRawImg = "66_5.bmp";
	string centerPointFile = "points.txt";

	//�ڶ������������ݳ�ʼ��
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//������������Rawͼ���Ӳ���Ŷ�mask���㣬ע���л��궨�壬ע�͵�SCENE_DEPTH_COMPUTE�Ķ��弴��
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//���Ĳ�������Matlab��������ӿ׾�ͼ�����Ⱦ��Ȼ���ӿ׾�ͼ���ӳ���ļ����ڵ�ǰĿ¼��
#else
	//���岽�����г�����ȵļ���,ע���л�ǰ��ĺ궨�壬ȡ��ע��SCENE_DEPTH_COMPUTE����

	string referSubImgName = "subAperatureImg.bmp";//�ӿ׾�ͼ��
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //�Ӳ��ļ�
	std::string referMaskXmlName = "confidentMatMask.xml"; //���Ŷ��ļ�
	string renderPointsMapping = "renderPointsMapping.txt";//��Ⱦ��ӳ���
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//������ ����ȫ���Ż�Matlab����
}


void ToolTwoTestDemo::data6compute()
{
	//��һ������Rawͼ�������ز���������
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x��y�����ϵ�ͼ��ƫ��ֵ
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; //��ȡ��ͼ����Ⱥ͸߶�

	int filterRadius = 8; //�˲��뾶
	float circleDiameter = 40.0; //СԲ����
	float circleNarrow = 1.5; //��������ֵ
	int dispMin = 6; //�Ӳ���Сֵ
	int dispMax = 10; //�Ӳ����ֵ
	float dispStep = 0.5; //�Ӳ����ֵ

	string folderName = "205_11_77";//����м������ļ���
	string inputRawImg = "205.bmp";
	string centerPointFile = "points.txt";

	//�ڶ������������ݳ�ʼ��
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//������������Rawͼ���Ӳ���Ŷ�mask���㣬ע���л��궨�壬ע�͵�SCENE_DEPTH_COMPUTE�Ķ��弴��
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//���Ĳ�������Matlab��������ӿ׾�ͼ�����Ⱦ��Ȼ���ӿ׾�ͼ���ӳ���ļ����ڵ�ǰĿ¼��
#else
	//���岽�����г�����ȵļ���,ע���л�ǰ��ĺ궨�壬ȡ��ע��SCENE_DEPTH_COMPUTE����

	string referSubImgName = "subAperatureImg.bmp";//�ӿ׾�ͼ��
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //�Ӳ��ļ�
	std::string referMaskXmlName = "confidentMatMask.xml"; //���Ŷ��ļ�
	string renderPointsMapping = "renderPointsMapping.txt";//��Ⱦ��ӳ���
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//������ ����ȫ���Ż�Matlab����
}

void ToolTwoTestDemo::data7compute()
{//���� 0321_6
	//��һ������Rawͼ�������ز���������
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x��y�����ϵ�ͼ��ƫ��ֵ
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; //��ȡ��ͼ����Ⱥ͸߶�

	int filterRadius = 8; //�˲��뾶
	float circleDiameter = 40.0; //СԲ����
	float circleNarrow = 1.5; //��������ֵ 
	/*
	int dispMin = 7; //�Ӳ���Сֵ
	int dispMax = 14; //�Ӳ����ֵ
	float dispStep = 0.5; //�Ӳ����ֵ
	*/
	int dispMin = 7; //�Ӳ���Сֵ
	int dispMax = 8; //�Ӳ����ֵ
	float dispStep = 0.5; //�Ӳ����ֵ


	string folderName = "205_5m";//����м������ļ���
	string inputRawImg = "205_5m.bmp";
	string centerPointFile = "points.txt";

	//�ڶ������������ݳ�ʼ��
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//������������Rawͼ���Ӳ���Ŷ�mask���㣬ע���л��궨�壬ע�͵�SCENE_DEPTH_COMPUTE�Ķ��弴��
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//���Ĳ�������Matlab��������ӿ׾�ͼ�����Ⱦ��Ȼ���ӿ׾�ͼ���ӳ���ļ����ڵ�ǰĿ¼��
#else
	//���岽�����г�����ȵļ���,ע���л�ǰ��ĺ궨�壬ȡ��ע��SCENE_DEPTH_COMPUTE����

	string referSubImgName = "subAperatureImg.bmp";//�ӿ׾�ͼ��
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //�Ӳ��ļ�
	std::string referMaskXmlName = "confidentMatMask.xml"; //���Ŷ��ļ�
	string renderPointsMapping = "renderPointsMapping.txt";//��Ⱦ��ӳ���
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//������ ����ȫ���Ż�Matlab����
}


void ToolTwoTestDemo::data8compute()
{//
	//��һ������Rawͼ�������ز���������
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x��y�����ϵ�ͼ��ƫ��ֵ
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; //��ȡ��ͼ����Ⱥ͸߶�

	int filterRadius = 8; //�˲��뾶
	float circleDiameter = 40.0; //СԲ����
	float circleNarrow = 1.5; //��������ֵ
	int dispMin = 8; //�Ӳ���Сֵ
	int dispMax = 14; //�Ӳ����ֵ
	float dispStep = 0.5; //�Ӳ����ֵ

	string folderName = "205_15m";//����м������ļ���
	string inputRawImg = "15.bmp";
	string centerPointFile = "points.txt";

	//�ڶ������������ݳ�ʼ��
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//������������Rawͼ���Ӳ���Ŷ�mask���㣬ע���л��궨�壬ע�͵�SCENE_DEPTH_COMPUTE�Ķ��弴��
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//���Ĳ�������Matlab��������ӿ׾�ͼ�����Ⱦ��Ȼ���ӿ׾�ͼ���ӳ���ļ����ڵ�ǰĿ¼��
#else
	//���岽�����г�����ȵļ���,ע���л�ǰ��ĺ궨�壬ȡ��ע��SCENE_DEPTH_COMPUTE����

	string referSubImgName = "subAperatureImg.bmp";//�ӿ׾�ͼ��
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //�Ӳ��ļ�
	std::string referMaskXmlName = "confidentMatMask.xml"; //���Ŷ��ļ�
	string renderPointsMapping = "renderPointsMapping.txt";//��Ⱦ��ӳ���
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//������ ����ȫ���Ż�Matlab����
}


void ToolTwoTestDemo::data9compute()
{//
	//��һ������Rawͼ�������ز���������
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x��y�����ϵ�ͼ��ƫ��ֵ
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; //��ȡ��ͼ����Ⱥ͸߶�

	int filterRadius = 8; //�˲��뾶
	float circleDiameter = 40.0; //СԲ����
	float circleNarrow = 1.5; //��������ֵ
	int dispMin = 7; //�Ӳ���Сֵ
	int dispMax = 15; //�Ӳ����ֵ
	float dispStep = 0.5; //�Ӳ����ֵ

	string folderName = "205_25m";//����м������ļ���
	string inputRawImg = "25.bmp";
	string centerPointFile = "points.txt";

	//�ڶ������������ݳ�ʼ��
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//������������Rawͼ���Ӳ���Ŷ�mask���㣬ע���л��궨�壬ע�͵�SCENE_DEPTH_COMPUTE�Ķ��弴��
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//���Ĳ�������Matlab��������ӿ׾�ͼ�����Ⱦ��Ȼ���ӿ׾�ͼ���ӳ���ļ����ڵ�ǰĿ¼��
#else
	//���岽�����г�����ȵļ���,ע���л�ǰ��ĺ궨�壬ȡ��ע��SCENE_DEPTH_COMPUTE����

	string referSubImgName = "subAperatureImg.bmp";//�ӿ׾�ͼ��
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //�Ӳ��ļ�
	std::string referMaskXmlName = "confidentMatMask.xml"; //���Ŷ��ļ�
	string renderPointsMapping = "renderPointsMapping.txt";//��Ⱦ��ӳ���
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//������ ����ȫ���Ż�Matlab����
}

void ToolTwoTestDemo::data10compute()
{//
	//��һ������Rawͼ�������ز���������
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1; //x��y�����ϵ�ͼ��ƫ��ֵ
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1; //��ȡ��ͼ����Ⱥ͸߶�

	int filterRadius = 8; //�˲��뾶
	float circleDiameter = 40.0; //СԲ����
	float circleNarrow = 1.5; //��������ֵ
	int dispMin = 7; //�Ӳ���Сֵ
	int dispMax = 15; //�Ӳ����ֵ
	float dispStep = 0.5; //�Ӳ����ֵ

	string folderName = "205_50m";//����м������ļ���
	string inputRawImg = "50.bmp";
	string centerPointFile = "points.txt";

	//�ڶ������������ݳ�ʼ��
	DepthComputeToolTwo depthComputeToolTwo;
	depthComputeToolTwo.parameterInit(folderName, centerPointFile, inputRawImg, yCenterStartOffset, xCenterStartOffset, yCenterEndOffset, xCenterEndOffset
		, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);

	//������������Rawͼ���Ӳ���Ŷ�mask���㣬ע���л��궨�壬ע�͵�SCENE_DEPTH_COMPUTE�Ķ��弴��
#ifndef SCENE_DEPTH_COMPUTE
	depthComputeToolTwo.rawImageDisparityCompute();

	//���Ĳ�������Matlab��������ӿ׾�ͼ�����Ⱦ��Ȼ���ӿ׾�ͼ���ӳ���ļ����ڵ�ǰĿ¼��
#else
	//���岽�����г�����ȵļ���,ע���л�ǰ��ĺ궨�壬ȡ��ע��SCENE_DEPTH_COMPUTE����

	string referSubImgName = "subAperatureImg.bmp";//�ӿ׾�ͼ��
	std::string referDispXmlName = "dispAfterSTCAAgainLocalSmooth.xml"; //�Ӳ��ļ�
	std::string referMaskXmlName = "confidentMatMask.xml"; //���Ŷ��ļ�
	string renderPointsMapping = "renderPointsMapping.txt";//��Ⱦ��ӳ���
	depthComputeToolTwo.sceneDepthCompute(referSubImgName, referDispXmlName, referMaskXmlName, renderPointsMapping);
#endif

	//������ ����ȫ���Ż�Matlab����
}