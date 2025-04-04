#include "STCA.h"
#include "StereoDisparity.h"
#include "StereoHelper.h"
#include "SegmentTree.h"

//
// Segment-Tree Cost Aggregation
//
void STCAaggreCV( const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol )
{
	//printf( "\n\t\tSegment Tree cost aggregation" );
	//printf( "\n\t\tCost volume need to be recompute" );

	int hei = lImg.rows;
	int wid = lImg.cols;
	float* pCV = NULL;
	// image format must convert
	Mat lSgImg, rSgImg;
	lImg.convertTo( lSgImg, CV_8U, 255 );
	//rImg.convertTo( rSgImg, CV_8U, 255 );
	cvtColor( lSgImg, lSgImg, COLOR_RGB2BGR );
	//cvtColor(rSgImg, rSgImg, COLOR_RGB2BGR);
	Mat sgLCost;
	CDisparityHelper dispHelper;
	// init segmentation tree cost volume
	sgLCost = Mat::zeros(1, hei * wid * maxDis, CV_32F);
	//CV_Assert(lSgImg.type() == CV_8UC3 && rSgImg.type() == CV_8UC3);
#ifdef RE_COMPUTE_COST
	// recompute cost volume
	sgLCost = 
		dispHelper.GetMatchingCost( lSgImg, rSgImg, maxDis );
#else
	// my cost to st
	// !!! mine start from 1
	// just used for cencus cost
	pCV = ( float* )sgLCost.data;
	for( int y = 0; y < hei; y ++ ) {
		for( int x = 0; x < wid; x ++ ) {
			for( int d = 0; d < maxDis; d ++ ) {
				float* cost = (float*)costVol[d].ptr<float>(y);
				*pCV = cost[ x ];
				pCV ++;
			}
		}
	}
#endif
	// build tree
	CSegmentTree stree;
	CColorWeight cWeight( lSgImg );
	stree.BuildSegmentTree( lSgImg, 0.1, 1200, cWeight);
	// filter cost volume
	stree.Filter(sgLCost, maxDis);

	// st cost to my
	pCV = ( float* )sgLCost.data;
	for( int y = 0; y < hei; y ++ ) {
		for( int x = 0; x < wid; x ++ ) {
			for( int d = 0; d < maxDis; d ++ ) {
				float* cost   = ( float* ) costVol[ d ].ptr<float>( y );
				cost[ x ]  = *pCV;
				pCV ++;
			}
		}
	}
}

