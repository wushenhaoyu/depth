/*!
 * \class ToolTwoTestDemo
 *
 * \brief ���ڷ�������ʾ��θ��������2.0���ݽ����㷨��ʹ��
 *
 * \author liuqian
 * \date һ�� 2018
 */

#ifndef __TOOLTWOTESTDEMO_H_
#define __TOOLTWOTESTDEMO_H_

#include <string> // 添加此行以包含 std::string 的定义

struct DepthParams {
	int xCenterStartOffset = 1;
	int yCenterStartOffset = 1;
	int xCenterEndOffset = 1;
	int yCenterEndOffset = 1;

	int filterRadius = 6;
	float circleDiameter = 44.0f;
	float circleNarrow = 1.5f;
	int dispMin = 5;
	int dispMax = 14;
	float dispStep = 0.25f;

	std::string folderName = "/home/jetson/Desktop/depth/205_66";
	std::string inputRawImg = "new.bmp";
	std::string centerPointFile = "points_new.txt";
};


class ToolTwoTestDemo
{


public:

	ToolTwoTestDemo();
	~ToolTwoTestDemo();
	DepthParams loadDepthParams(const std::string& path);
	void data1compute();
	void data2compute();
	void data3compute();
	void data4compute();
	void data5compute();

	void data6compute();
	void data7compute();
	void data8compute();
	void data9compute();
	void data10compute();
	
private:

};


#endif