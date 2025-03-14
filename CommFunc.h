/*!
 * \file CommFunc.h
 *
 * \author liuqian
 * \date 十一月 2017
 *
 * 
 */

#ifndef __COMMFUNC_H_
#define __COMMFUNC_H_

#include <string>
#include <iostream>
#include <bitset>
#include <fstream>
#include <stdio.h>
#include <omp.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <float.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>  
//#include <opencv2/nonfree/nonfree.hpp>  
//#include <opencv2/legacy/legacy.hpp>  

using namespace std;
using namespace cv;

/*颜色与梯度加权的系数调整*/
#define TAU_1 0.02745   //0.02745
#define TAU_2 0.3//0.00784  0.4
#define ALPHA 0.11
#define DOUBLE_MAX 1e10
#define PI 3.14159265f //pi的值
#define BASELINES_NUM 6 //基线个数
#define SEARCH_DIRECTION 8 //搜索方向，8邻域搜索
#define TH 40 //设置梯度的方向阈值

enum class FilterOptimizeKind{
	e_boxca, e_bfca, e_gfca, e_stca, e_nlcca
};//box滤波，双边滤波,导向滤波,ST局部优化，NLC局部优化


#endif
