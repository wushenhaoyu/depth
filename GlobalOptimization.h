/*!
 * \file GlobalOptimization.h
 *
 * \author liuqian
 * \date ʮһ�� 2016
 *
 * ȫ���Ż�������ͼ���㷨
 */

#ifndef __GLOBAL_OPTIMIZATION_H__
#define __GLOBAL_OPTIMIZATION_H__

#include "CommFunc.h"
#include "gcov3/GCoptimization.h"

int fnCost(int pix1, int pix2, int i, int j);//ƽ������

void globalOptimize(const int width, const int height, const int num_pixels, const int num_labels, Mat *&costVol, Mat &disp);//ȫ���Ż�����

int* generateDataFunction(const int width, const int height, Mat *&costVol);//����������

void getDisparities(const int width, const int height, int *resDisp, Mat &disp);//��ȡ�Ӳ�ͼ

#endif