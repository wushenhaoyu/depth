#include "GlobalOptimization.h"


static int numLabels = 0;
static int g_imageWidth = 0;
// const int LAMBDA = 800;
// const int SMOOTHNESS_TRUNC = 3;

const int LAMBDA = 1200;
const int SMOOTHNESS_TRUNC = 3;

int fnCost(int pix1, int pix2, int i, int j)
{//ƽ������
// 	if (abs(pix1 - pix2) <= 1 || abs(pix1 / g_imageWidth - pix2 / g_imageWidth) <= 1 || (abs(pix1 - pix2) >= numLabels - 1 && abs(pix1 - pix2) <= numLabels + 1))
// 	{
// 		int d = i - j;
// 		return LAMBDA*std::min(abs(d), SMOOTHNESS_TRUNC);
// 	}
// 	else
// 		return 0;
	if (abs(pix1 - pix2) <= 1 || (abs(pix1 / g_imageWidth - pix2 / g_imageWidth) <= 1 && abs(pix1 % g_imageWidth - pix2 % g_imageWidth)<=1 ) || (abs(pix1 - pix2) >= numLabels - 1 && abs(pix1 - pix2) <= numLabels + 1))
	{
		int d = i - j;
		return LAMBDA*std::min(d*d, SMOOTHNESS_TRUNC);
	}
	else
		return 0;
}

void getDisparities(const int width, const int height, int *resDisp, Mat &disp)
{//��ȡ�Ӳ�ͼ
	int n = 0;
	for (int y = 0; y < height; y++)
	{
		uchar* lDisData = (uchar*)disp.ptr<uchar>(y);
		for (int x = 0; x < width; x++)
		{
			lDisData[x] = resDisp[n++];
		}
	}
}

int* generateDataFunction(const int width, const int height, Mat *&costVol)
{//����������
	double minVal = FLT_MAX, maxVal = 0.0;
	double tempMinVal, tempMaxVal;
	for (int i = 0; i < numLabels; i++)
	{
		minMaxLoc(costVol[i], &tempMinVal, &tempMaxVal);
		minVal = std::min(minVal, tempMinVal);
		maxVal = std::max(maxVal, tempMaxVal);
	} //local smooth: 0.004217~0.009965
	//guide smooth : 0.003758~0.0104
	//STCA smooth : 0.06557~2.141437

	int *D = new int[width*height*numLabels];//ȫ���Ż������������
	int dataCount = 0;
//#pragma omp parallel for
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			for (int d = 0; d < numLabels; d++)
			{
				float* costData = (float*)costVol[d].ptr<float>(y);
				//D[dataCount++] = int(std::min(costData[x] * 10000, maxVal * 10000 * 0.9));/////////////////test   
				//D[dataCount++] = int(std::min(costData[x] * 10000.0, maxVal * 10000 * 0.75));/////////////////test  ////////////////
				D[(y*width + x)*numLabels + d] = int(std::min(costData[x] * 10000.0, maxVal * 10000 * 0.75));
				//D[dataCount++] = int(costData[x] * 10000);/////////////////test  **********************
				//D[dataCount++] = int(costData[x] * 1000000);/////////////////test
				//D[dataCount++] = std::min((costData[x] - minVal)*(255.0 - 0.0) / (maxVal - minVal) + 0.0, DATACOST_THRES);
			}
		}
	}

	return D;
}

void globalOptimize(const int width, const int height, const int num_pixels, const int num_labels, Mat *&costVol, Mat &disp)
{//ȫ���Ż�����
	numLabels = num_labels;
	g_imageWidth = width;

	int *costData = generateDataFunction(width, height, costVol);
	int *resultDisp = new int[num_pixels];

	//GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width, height, num_labels);
	GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels, num_labels);
	gc->setDataCost(costData);
	gc->setSmoothCost(&fnCost);

	
	for (int y = 0; y < height; y++)
		for (int x = 1; x < width; x++)
			gc->setNeighbors(x + y*width, x - 1 + y*width);

	// next set up vertical neighbors
	for (int y = 2; y < height; y++)
		for (int x = 0; x < width; x++)
			gc->setNeighbors(x + y*width, x + (y - 1)*width, 2);

	printf("\nBefore optimization energy is %ld, the data cost is %ld, the smooth cost is %ld \n", gc->compute_energy(),gc->giveDataEnergy(),gc->giveSmoothEnergy());
	//gc->expansion(10);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
	gc->swap(2);
	printf("\nBefore optimization energy is %ld, the data cost is %ld, the smooth cost is %ld \n", gc->compute_energy(), gc->giveDataEnergy(), gc->giveSmoothEnergy());

	for (int i = 0; i < num_pixels; i++)
		resultDisp[i] = gc->whatLabel(i);

	getDisparities(width, height, resultDisp, disp);

	delete[]costData;
	delete[]resultDisp;
}