/*!
 * \class DisparityRefinement
 *
 * \brief ��dataCost��������һ��ȫ�ֻ�ֲ��Ż�
 *
 * \author liuqian
 * \date ʮһ�� 2016
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
	void localFilterOrOptimize(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol, FilterOptimizeKind curFilterOptimizeKind);//���оֲ��˲������Ż�
	void globalGCOptimize(const Mat& lImg, const int num_labels, Mat *&costVol, Mat &disp);//��ͼ�����ȫ���Ż�
protected:
	void _BFCAaggreCV(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol);//˫���˲�
	void _BoxCAaggreCV(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol);//�����˲�
	void _GFCAaggreCV(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol);//�����˲�
	void _STCAaggreCV(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol);//ST�ֲ��Ż�
	void _NLCCAaggreCV(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol);//NLCC�ֲ��Ż�

	// cum sum like cumsum in matlab
	Mat CumSum(const Mat& src, const int d);

	//  %   BOXFILTER   O(1) time box filtering using cumulative sum
	//	%
	//	%   - Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
	//  %   - Running time independent of r; 
	//  %   - Equivalent to the function: colfilt(imSrc, [2*r+1, 2*r+1], 'sliding', @sum);
	//  %   - But much faster.
	Mat BoxFilter(const Mat& imSrc, const int r = 9);//�����˲�
	//  %   GUIDEDFILTER   O(1) time implementation of guided filter.
	//	%
	//	%   - guidance image: I (should be a gray-scale/single channel image)
	//	%   - filtering input image: p (should be a gray-scale/single channel image)
	//	%   - local window radius: r
	//	%   - regularization parameter: eps
	Mat GuidedFilter(const Mat& I, const Mat& p, const int r = 9, const float eps = 0.0001);//�����˲�

	Mat BilateralFilter(const Mat& I, const Mat& p, const int wndSZ = 9, double sig_sp = 4.5, const double sig_clr = 0.03);//˫���˲�

	int fnCost(int pix1, int pix2, int i, int j);

	int* generateDataFunction(int width, int height, Mat *&costVol);

	void getDisparities(int width, int height, int *resDisp, Mat &disp);
private:

	DisparityRefinement();
	int m_numLabels;
	static DisparityRefinement* m_pDisparityRefinement;//��̬��ͼ������
};


#endif