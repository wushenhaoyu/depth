#include "DisparityRefinement.h"

DisparityRefinement* DisparityRefinement::m_pDisparityRefinement = nullptr;

DisparityRefinement::DisparityRefinement(){

}

DisparityRefinement::~DisparityRefinement(){

}

DisparityRefinement* DisparityRefinement::getInstance()
{
	if (nullptr == m_pDisparityRefinement){
		m_pDisparityRefinement = new DisparityRefinement();
	}
	return m_pDisparityRefinement;
}

void DisparityRefinement::localFilterOrOptimize(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol, FilterOptimizeKind curFilterOptimizeKind)
{//进行局部滤波或者优化
	switch (curFilterOptimizeKind)
	{
	case FilterOptimizeKind::e_boxca:
		_BoxCAaggreCV(lImg, rImg, maxDis, costVol);
		break;
	case FilterOptimizeKind::e_bfca:
		_BFCAaggreCV(lImg, rImg, maxDis, costVol);
		break;
	case FilterOptimizeKind::e_gfca:
		_GFCAaggreCV(lImg, rImg, maxDis, costVol);
		break;
	case FilterOptimizeKind::e_stca:
		_STCAaggreCV(lImg, rImg, maxDis, costVol);
		break;
	case FilterOptimizeKind::e_nlcca:
		_NLCCAaggreCV(lImg, rImg, maxDis, costVol);
		break;
	default:
		break;
	}
}

void DisparityRefinement::_BFCAaggreCV(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol)
{//双边滤波
	// filtering cost volume
	for (int d = 1; d < maxDis; d++) {
		printf("-c-a");
		// change BF window size for KITTI
		costVol[d] = BilateralFilter(lImg, costVol[d], 35);
	}
}
///////////////////////////////////////////////////////////////////////////////////////

void DisparityRefinement::_BoxCAaggreCV(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol)
{//方框滤波
	// filtering cost volume
	for (int d = 1; d < maxDis; d++) {
		printf("-c-a");
		// 7 x 7 box filter
		costVol[d] = BoxFilter(costVol[d], 3);
	}
}
/////////////////////////////////////////////////////////////////////////////////////////
void DisparityRefinement::_GFCAaggreCV(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol)
{//导向滤波
	// filtering cost volume
	for (int d = 1; d < maxDis; d++) {
		printf("-c-a");
		costVol[d] = GuidedFilter(lImg, costVol[d]);
	}
}

void DisparityRefinement::_STCAaggreCV(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol)
{//ST局部优化
	STCAaggreCV(lImg, rImg, maxDis, costVol);
}

void DisparityRefinement::_NLCCAaggreCV(const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol)
{//NLCC局部优化
	NLCCAaggreCV(lImg, rImg, maxDis, costVol);
}

/*************************GuidedFilter**************************************/

// cum sum like cumsum in matlab
Mat DisparityRefinement::CumSum(const Mat& src, const int d)
{
	int H = src.rows;
	int W = src.cols;

	int itype = src.type();
	Mat dest = Mat::zeros(H, W, src.type());
	//Mat dest = Mat::zeros(H, W, CV_64FC1);

	if (d == 1) {
		// summation over column
		for (int y = 0; y < H; y++) {
			double* curData = (double*)dest.ptr<double>(y);
			double* preData = (double*)dest.ptr<double>(y);
			if (y) {
				// not first row
				preData = (double*)dest.ptr<double>(y - 1);
			}
			double* srcData = (double*)src.ptr<double>(y);
			for (int x = 0; x < W; x++) {
				curData[x] = preData[x] + srcData[x];
			}
		}
	}
	else {
		// summation over row
		for (int y = 0; y < H; y++) {
			double* curData = (double*)dest.ptr<double>(y);
			double* srcData = (double*)src.ptr<double>(y);
			for (int x = 0; x < W; x++) {
				if (x) {
					curData[x] = curData[x - 1] + srcData[x];
				}
				else {
					curData[x] = srcData[x];
				}
			}
		}
	}
	return dest;
}
//  %   BOXFILTER   O(1) time box filtering using cumulative sum
//	%
//	%   - Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
//  %   - Running time independent of r; 
//  %   - Equivalent to the function: colfilt(imSrc, [2*r+1, 2*r+1], 'sliding', @sum);
//  %   - But much faster.
Mat DisparityRefinement::BoxFilter(const Mat& imSrc, const int r)
{
	int H = imSrc.rows;
	int W = imSrc.cols;
	// image size must large than filter size
	CV_Assert(W >= r && H >= r);
	Mat imDst = Mat::zeros(H, W, imSrc.type());
	//Mat imDst = Mat::zeros(H, W, CV_64FC1);

	// cumulative sum over Y axis
	Mat imCum = CumSum(imSrc, 1);
	// difference along Y ( [ 0, r ], [r + 1, H - r - 1], [ H - r, H ] )
	for (int y = 0; y < r + 1; y++) {
		double* dstData = (double*)imDst.ptr<double>(y);
		double* plusData = (double*)imCum.ptr<double>(y + r);
		for (int x = 0; x < W; x++) {
			dstData[x] = plusData[x];
		}

	}
	for (int y = r + 1; y < H - r; y++) {
		double* dstData = (double*)imDst.ptr<double>(y);
		double* minusData = (double*)imCum.ptr<double>(y - r - 1);
		double* plusData = (double*)imCum.ptr<double>(y + r);
		for (int x = 0; x < W; x++) {
			dstData[x] = plusData[x] - minusData[x];
		}
	}
	for (int y = H - r; y < H; y++) {
		double* dstData = (double*)imDst.ptr<double>(y);
		double* minusData = (double*)imCum.ptr<double>(y - r - 1);
		double* plusData = (double*)imCum.ptr<double>(H - 1);
		for (int x = 0; x < W; x++) {
			dstData[x] = plusData[x] - minusData[x];
		}
	}

	// cumulative sum over X axis
	imCum = CumSum(imDst, 2);
	for (int y = 0; y < H; y++) {
		double* dstData = (double*)imDst.ptr<double>(y);
		double* cumData = (double*)imCum.ptr<double>(y);
		for (int x = 0; x < r + 1; x++) {
			dstData[x] = cumData[x + r];
		}
		for (int x = r + 1; x < W - r; x++) {
			dstData[x] = cumData[x + r] - cumData[x - r - 1];
		}
		for (int x = W - r; x < W; x++) {
			dstData[x] = cumData[W - 1] - cumData[x - r - 1];
		}
	}
	return imDst;
}
//  %   GUIDEDFILTER   O(1) time implementation of guided filter.
//	%
//	%   - guidance image: I (should be a gray-scale/single channel image)
//	%   - filtering input image: p (should be a gray-scale/single channel image)
//	%   - local window radius: r
//	%   - regularization parameter: eps


Mat DisparityRefinement::GuidedFilter(const Mat& I, const Mat& p, const int r, const float eps)
{
	// filter signal must be 1 channel
	CV_Assert(p.type() == CV_64FC1);

	//int itype = I.type();

	int H = I.rows;
	int W = I.cols;
	Mat N = Mat::ones(H, W, CV_64FC1);
	N = BoxFilter(N, r);

	if (1 == I.channels()) {
		// gray guidence
		Mat mean_I = BoxFilter(I, r) / N;
		Mat mean_p = BoxFilter(p, r) / N;
		Mat tmp;

		multiply(I, p, tmp);
		Mat mean_Ip = BoxFilter(tmp, r) / N;

		multiply(mean_I, mean_p, tmp);
		Mat cov_Ip = mean_Ip - tmp;

		multiply(I, I, tmp);
		Mat mean_II = BoxFilter(tmp, r) / N;

		multiply(mean_I, mean_I, tmp);
		Mat var_I = mean_II - tmp;

		Mat a = cov_Ip / (var_I + eps);

		multiply(a, mean_I, tmp);
		Mat b = mean_p - tmp;

		Mat mean_a = BoxFilter(a, r) / N;
		Mat mean_b = BoxFilter(b, r) / N;

		multiply(mean_a, I, tmp);
		Mat q = tmp + mean_b;
		return q;
	}
	else {
		// color guidence
		// image must in RGB format!!!

		Mat rgb[3];
		// 		for (int c = 0; c < 3; c++) {
		// 			rgb[c] = Mat::zeros(H, W, CV_64FC1);
		// 		}

		split(I, rgb);
		Mat mean_I[3];
		for (int c = 0; c < 3; c++) {
			mean_I[c] = BoxFilter(rgb[c], r) / N;
		}
		Mat mean_p = BoxFilter(p, r) / N;
		Mat tmp;
		Mat mean_Ip[3];
		for (int c = 0; c < 3; c++) {
			//int itype = rgb[c].type();
			//int itype2 = p.type();

			multiply(rgb[c], p, tmp);
			mean_Ip[c] = BoxFilter(tmp, r) / N;
		}
		/*% covariance of (I, p) in each local patch.*/
		Mat cov_Ip[3];
		for (int c = 0; c < 3; c++) {
			multiply(mean_I[c], mean_p, tmp);
			cov_Ip[c] = mean_Ip[c] - tmp;
		}
		//  % variance of I in each local patch: the matrix Sigma in Eqn (14).
		//	% Note the variance in each local patch is a 3x3 symmetric matrix:
		//  %           rr, rg, rb
		//	%   Sigma = rg, gg, gb
		//	%           rb, gb, bb
		Mat var_I[6];
		int varIdx = 0;
		for (int c = 0; c < 3; c++) {
			for (int c_p = c; c_p < 3; c_p++) {
				multiply(rgb[c], rgb[c_p], tmp);
				var_I[varIdx] = BoxFilter(tmp, r) / N;
				multiply(mean_I[c], mean_I[c_p], tmp);
				var_I[varIdx] -= tmp;
				varIdx++;
			}
		}
		Mat a[3];
		for (int c = 0; c < 3; c++) {
			a[c] = Mat::zeros(H, W, CV_64FC1);
		}
		Mat epsEye = Mat::eye(3, 3, CV_64FC1);
		epsEye *= eps;
#ifdef _DEBUG
		double duration;
		duration = static_cast<double>(cv::getTickCount());
#endif
		for (int y = 0; y < H; y++) {
			double* vData[6];
			for (int v = 0; v < 6; v++) {
				vData[v] = (double*)var_I[v].ptr<double>(y);
			}
			double* cData[3];
			for (int c = 0; c < 3; c++) {
				cData[c] = (double *)cov_Ip[c].ptr<double>(y);
			}
			double* aData[3];
			for (int c = 0; c < 3; c++) {
				aData[c] = (double*)a[c].ptr<double>(y);
			}
			for (int x = 0; x < W; x++) {
#ifndef FAST_INV
				Mat sigma = (Mat_<double>(3, 3) <<
					vData[0][x], vData[1][x], vData[2][x],
					vData[1][x], vData[3][x], vData[4][x],
					vData[2][x], vData[4][x], vData[5][x]
					);
				sigma += epsEye;
				Mat cov_Ip_13 = (Mat_<double>(1, 3) <<
					cData[0][x], cData[1][x], cData[2][x]);
				tmp = cov_Ip_13 * sigma.inv();
				double* tmpData = tmp.ptr<double>(0);
				for (int c = 0; c < 3; c++) {
					aData[c][x] = tmpData[c];
				}
#else
				double c0 = cData[0][x];
				double c1 = cData[1][x];
				double c2 = cData[2][x];
				double a11 = vData[0][x] + eps;
				double a12 = vData[1][x];
				double a13 = vData[2][x];
				double a21 = vData[1][x];
				double a22 = vData[3][x] + eps;
				double a23 = vData[4][x];
				double a31 = vData[2][x];
				double a32 = vData[4][x];
				double a33 = vData[5][x] + eps;
				double DET = a11 * (a33 * a22 - a32 * a23) -
					a21 * (a33 * a12 - a32 * a13) +
					a31 * (a23 * a12 - a22 * a13);
				DET = 1 / DET;
				aData[0][x] = DET * (
					c0 * (a33 * a22 - a32 * a23) +
					c1 * (a31 * a23 - a33 * a21) +
					c2 * (a32 * a21 - a31 * a22)
					);
				aData[1][x] = DET * (
					c0 * (a32 * a13 - a33 * a12) +
					c1 * (a33 * a11 - a31 * a13) +
					c2 * (a31 * a12 - a32 * a11)
					);
				aData[2][x] = DET * (
					c0 * (a23 * a12 - a22 * a13) +
					c1 * (a21 * a13 - a23 * a11) +
					c2 * (a22 * a11 - a21 * a12)
					);
#endif
			}
		}
#ifdef _DEBUG
		duration = static_cast<double>(cv::getTickCount()) - duration;
		duration /= cv::getTickFrequency(); // the elapsed time in sec
		// printf( "Inter Time: %lf s\n", duration );
#endif
		Mat b = mean_p.clone();
		for (int c = 0; c < 3; c++) {
			multiply(a[c], mean_I[c], tmp);
			b -= tmp;
		}
		Mat q = BoxFilter(b, r);
		for (int c = 0; c < 3; c++) {
			multiply(BoxFilter(a[c], r), rgb[c], tmp);
			q += tmp;
		}
		q /= N;
		return q;
	}
}

/*****************************************BilateralFilter**********************************************************************************/

Mat DisparityRefinement::BilateralFilter(const Mat& I, const Mat& p, const int wndSZ /*= 9*/, double sig_sp /*= 5*/, const double sig_clr /*= 0.028 */)
{
	// filter signal must be 1 channel
	CV_Assert(p.type() == CV_64FC1);

	// range parameter is half window size
	sig_sp = wndSZ / 2.0f;

	int H = I.rows;
	int W = I.cols;
	int H_WD = wndSZ / 2;
	Mat dest = p.clone();
	if (1 == I.channels()) {
		for (int y = 0; y < H; y++) {
			double* pData = (double*)(p.ptr<double>(y));
			double* destData = (double*)(dest.ptr<double>(y));
			double* IP = (double*)(I.ptr<double>(y));
			for (int x = 0; x < W; x++) {
				double sum = 0.0f;
				double sumWgt = 0.0f;
				for (int wy = -H_WD; wy <= H_WD; wy++) {
					int qy = y + wy;
					if (qy < 0) {
						qy += H;
					}
					if (qy >= H) {
						qy -= H;
					}
					double* qData = (double*)(p.ptr<double>(qy));
					double* IQ = (double*)(I.ptr<double>(qy));
					for (int wx = -H_WD; wx <= H_WD; wx++) {
						int qx = x + wx;
						if (qx < 0) {
							qx += W;
						}
						if (qx >= W) {
							qx -= W;
						}
						double spDis = wx * wx + wy * wy;
						double clrDis = fabs(IQ[qx] - IP[x]);
						double wgt = exp(-spDis / (sig_sp * sig_sp) - clrDis * clrDis / (sig_clr * sig_clr));
						sum += wgt * qData[qx];
						sumWgt += wgt;
					}
				}
				destData[x] = sum / sumWgt;
			}
		}
	}
	else if (3 == I.channels()) {
		for (int y = 0; y < H; y++) {
			double* pData = (double*)(p.ptr<double>(y));
			double* destData = (double*)(dest.ptr<double>(y));
			double* IP = (double*)(I.ptr<double>(y));
			for (int x = 0; x < W; x++) {
				double* pClr = IP + 3 * x;
				double sum = 0.0f;
				double sumWgt = 0.0f;
				for (int wy = -H_WD; wy <= H_WD; wy++) {
					int qy = y + wy;
					if (qy < 0) {
						qy += H;
					}
					if (qy >= H) {
						qy -= H;
					}
					double* qData = (double*)(p.ptr<double>(qy));
					double* IQ = (double*)(I.ptr<double>(qy));
					for (int wx = -H_WD; wx <= H_WD; wx++) {
						int qx = x + wx;
						if (qx < 0) {
							qx += W;
						}
						if (qx >= W) {
							qx -= W;
						}
						double* qClr = IQ + 3 * qx;
						double spDis = wx * wx + wy * wy;
						double clrDis = 0.0f;
						for (int c = 0; c < 3; c++) {
							clrDis += fabs(pClr[c] - qClr[c]);
						}
						clrDis *= 0.333333333;
						double wgt = exp(-spDis / (sig_sp * sig_sp) - clrDis * clrDis / (sig_clr * sig_clr));
						sum += wgt * qData[qx];
						sumWgt += wgt;
					}
				}
				destData[x] = sum / sumWgt;
			}
		}
	}
	return dest;
}

/*****************************************************************全局优化部分******************************************************************************/

void DisparityRefinement::globalGCOptimize(const Mat& lImg, const int num_labels, Mat *&costVol, Mat &disp)
{//用图割进行全局优化
	globalOptimize(lImg.cols, lImg.rows, lImg.cols*lImg.rows, num_labels, costVol, disp);
}