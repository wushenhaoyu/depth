/*!
 * \class 
 *
 * \brief 数据存储类，可以对视差数据或者datacost数据进行存储或者显示，便于后续扩充功能
 *
 * \author liuqian
 * \date 一月 2018
 */

#ifndef __DATADEAL_H_
#define __DATADEAL_H_

#include "CommFunc.h"

class DataDeal
{
public:
	DataDeal();
	~DataDeal();

	void readDataCostFromXML(std::string dataCostFileName, cv::Mat *&costVol); //从xml文件中读取视差dataCost数据
	void storeDataCostToXML(std::string dataCostFileName, const cv::Mat *&costVol, int imgHeight, int imgWidth, int maxDis); //存储视差dataCost数据到xml文件中
	void WTAMatch(cv::Mat *&costVol, cv::Mat &disMap, int maxDis); //从视差dataCost数据中得到视差数据，用WTA算法
	void dispMapShow(std::string dispImgName, const cv::Mat &disMap); //显示并存储视差图像
	void dispMapShowForColor(std::string dispImgName, const cv::Mat &disMap); //显示视差图，以彩色编码显示并存储
	void readDispMapFromXML(std::string dispFileName, cv::Mat &disMap); //读取视差图信息
	void storeDispMapToXML(std::string dispFileName, cv::Mat &disMap); //存储对应的视差状态信息
private:

};

#endif