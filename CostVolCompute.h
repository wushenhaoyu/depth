#ifndef __COSTVOLCOMPUTE_H_
#define __COSTVOLCOMPUTE_H_

#include "CommFunc.h"
#include <opencv2/opencv.hpp>

class DataParameter;
struct RawImageParameter;
struct MicroImageParameter;
struct DisparityParameter;

class CostVolCompute
{
public:
    CostVolCompute();
    ~CostVolCompute();

    void costVolDataCompute(const DataParameter &dataParameter, cv::Mat *costVol); // 计算 Raw 图像对应的 dataCost

    void costVolDataCompute(cv::Mat *costVol, int y, int x, const RawImageParameter &rawImageParameter,
        const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter,
        const cv::Mat &inputImg, const cv::Mat &gradImg); // 更新声明


    float costVolDataCompute(int y, int x, int py, int px, int d, const RawImageParameter &rawImageParameter,
        const MicroImageParameter &microImageParameter, const DisparityParameter &disparityParameter,
        const cv::Mat &inputImg, const cv::Mat &gradImg); // 更新声明

    bool isCurPointValid(cv::Point2d &matchPoint, int matchCenterIndex, const RawImageParameter &rawImageParameter,
        const MicroImageParameter &microImageParameter); // 判断当前位置的匹配点是否有效

    float bilinearInsertValue(const cv::Point2d &curPoint, const cv::Point2d &matchPoint, const cv::Mat &inputImg, const cv::Mat &gradImg); // 计算当前点与匹配点的 cost 值

    inline float myCostGrd(float* lC, float* rC, float* lG, float* rG); // 颜色和梯度进行加权

    inline float myCostGrd(float* lC, float* lG); // 颜色和梯度加权（边界处理）

private:
    cv::Mat m_inputImg; // 存储原始图像
    cv::Mat m_gradImg;  // 原始图像对应的梯度图像
};

#endif