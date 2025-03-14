/*!
 * \class DisparityRefinement
 *
 * \brief 对dataCost矩阵做进一步全局或局部优化
 *
 * \author liuqian
 * \date 十一月 2016
 */

#ifndef __DISPARITYREFINEMENT_H_
#define __DISPARITYREFINEMENT_H_

#include "CommFunc.h"
#include "CANLC/NLCCA.h"
#include "CAST/STCA.h"
#include "GlobalOptimization.h"

#define FAST_INV

class DisparityRefinement
{
public:
	
	virtual ~DisparityRefinement();
	static DisparityRefinement* getInstance();
	void localFilterOrOptimize(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol, FilterOptimizeKind curFilterOptimizeKind);//进行局部滤波或者优化
	void globalGCOptimize(const Mat& lImg, const int num_labels, Mat *&costVol, Mat &disp);//用图割进行全局优化
protected:
	void _BFCAaggreCV(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol);//双边滤波
	void _BoxCAaggreCV(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol);//方框滤波
	void _GFCAaggreCV(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol);//导向滤波
	void _STCAaggreCV(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol);//ST局部优化
	void _NLCCAaggreCV(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol);//NLCC局部优化

	// cum sum like cumsum in matlab
	Mat CumSum(const Mat& src, const int d);

	//  %   BOXFILTER   O(1) time box filtering using cumulative sum
	//	%
	//	%   - Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
	//  %   - Running time independent of r; 
	//  %   - Equivalent to the function: colfilt(imSrc, [2*r+1, 2*r+1], 'sliding', @sum);
	//  %   - But much faster.
	Mat BoxFilter(const Mat& imSrc, const int r = 9);//方框滤波
	//  %   GUIDEDFILTER   O(1) time implementation of guided filter.
	//	%
	//	%   - guidance image: I (should be a gray-scale/single channel image)
	//	%   - filtering input image: p (should be a gray-scale/single channel image)
	//	%   - local window radius: r
	//	%   - regularization parameter: eps
	Mat GuidedFilter(const Mat& I, const Mat& p, const int r = 9, const float eps = 0.0001);//导向滤波

	Mat BilateralFilter(const Mat& I, const Mat& p, const int wndSZ = 9, double sig_sp = 4.5, const double sig_clr = 0.03);//双边滤波

	int fnCost(int pix1, int pix2, int i, int j);

	int* generateDataFunction(int width, int height, Mat *&costVol);

	void getDisparities(int width, int height, int *resDisp, Mat &disp);
private:

	DisparityRefinement();
	int m_numLabels;
	static DisparityRefinement* m_pDisparityRefinement;//静态地图管理器
};


#endif