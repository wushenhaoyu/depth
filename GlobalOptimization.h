/*!
 * \file GlobalOptimization.h
 *
 * \author liuqian
 * \date 十一月 2016
 *
 * 全局优化函数：图割算法
 */

#ifndef __GLOBAL_OPTIMIZATION_H__
#define __GLOBAL_OPTIMIZATION_H__

#include "CommFunc.h"
#include "gcov3/GCoptimization.h"

int fnCost(int pix1, int pix2, int i, int j);//平滑函数

void globalOptimize(const int width, const int height, const int num_pixels, const int num_labels, Mat *&costVol, Mat &disp);//全局优化函数

int* generateDataFunction(const int width, const int height, Mat *&costVol);//生成数据项

void getDisparities(const int width, const int height, int *resDisp, Mat &disp);//获取视差图

#endif