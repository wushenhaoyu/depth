/*!
 * \class 
 *
 * \brief ���ݴ洢�࣬���Զ��Ӳ����ݻ���datacost���ݽ��д洢������ʾ�����ں������书��
 *
 * \author liuqian
 * \date һ�� 2018
 */

#ifndef __DATADEAL_H_
#define __DATADEAL_H_

#include "CommFunc.h"

class DataDeal
{
public:
	DataDeal();
	~DataDeal();

	void readDataCostFromXML(std::string dataCostFileName, cv::Mat *&costVol); //��xml�ļ��ж�ȡ�Ӳ�dataCost����
	void storeDataCostToXML(std::string dataCostFileName, const cv::Mat *&costVol, int imgHeight, int imgWidth, int maxDis); //�洢�Ӳ�dataCost���ݵ�xml�ļ���
	void WTAMatch(int width,  int height,  int maxDis); //���Ӳ�dataCost�����еõ��Ӳ����ݣ���WTA�㷨
	void WTAMatch1(int width,  int height,  int maxDis); //���Ӳ�dataCost�����еõ��Ӳ����ݣ���WTA�㷨
	void WTAMatch2(int width,  int height,  int maxDis); //���Ӳ�dataCost�����еõ��Ӳ����ݣ���WTA�㷨
	void dispMapShow(std::string dispImgName, const cv::Mat &disMap); //��ʾ���洢�Ӳ�ͼ��
	void dispMapShowForColor(std::string dispImgName, const cv::Mat &disMap); //��ʾ�Ӳ�ͼ���Բ�ɫ������ʾ���洢
	void readDispMapFromXML(std::string dispFileName, cv::Mat &disMap); //��ȡ�Ӳ�ͼ��Ϣ
	void storeDispMapToXML(std::string dispFileName, cv::Mat &disMap); //�洢��Ӧ���Ӳ�״̬��Ϣ
private:

};

#endif