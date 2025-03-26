#include "DataDeal.h"
#include "DataParameter.cuh"
using namespace std;
using namespace cv;

DataDeal::DataDeal()
{

}

DataDeal::~DataDeal()
{

}

void DataDeal::readDataCostFromXML(std::string dataCostFileName, cv::Mat *&costVol)
{//��xml�ļ��ж�ȡ�Ӳ�dataCost����
	FileStorage fs;
	char strName[50] = { "\0" };
	fs.open(dataCostFileName, FileStorage::READ);
	int labelNum = 0, height = 0, width = 0;
	fs["height"] >> height;
	fs["width"] >> width;
	fs["labelNum"] >> labelNum;
	Mat readMat = Mat(height, width, CV_32FC1);
	for (int i = 0; i < labelNum; i++)
	{
		sprintf(strName, "dataCostSet%d", i);
		fs[strName] >> readMat;
		costVol[i] = readMat.clone();
	}
	fs.release();
	cout << "read dataCost over!\n";
}

void DataDeal::storeDataCostToXML(std::string dataCostFileName, const cv::Mat *&costVol, int imgHeight, int imgWidth, int maxDis)
{//�洢�Ӳ�dataCost���ݵ�xml�ļ���
	FileStorage fs;
	vector<Mat> dispVec;
	fs.open(dataCostFileName, FileStorage::WRITE);
	char strName[50] = { "\0" };
	fs << "height" << imgHeight << "width" << imgWidth << "labelNum" << maxDis;
	for (int i = 0; i < maxDis; i++)
	{
		sprintf(strName, "dataCostSet%d", i);
		fs << strName << costVol[i];
	}
	fs.release();
}

extern __device__ float* d_costVol;
extern __device__ float* d_rawDisp;
extern __constant__ DisparityParameter d_disparityParameter;
extern __constant__ RawImageParameter d_rawImageParameter;

__global__ void wtamatchKernel()
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
	int height = d_rawImageParameter.m_recImgHeight;
	int width = d_rawImageParameter.m_recImgWidth;
    // 确保线程在图像范围内
    if (y < height && x < width)
    {
        float minCost = FLT_MAX;
        int minDis = 0;

        for (int d = 1; d < d_disparityParameter.m_dispMax; d++) {
            // 计算该视差下的代价值
            float* costData = &d_costVol[d * height * width + y * width + x];

            // 若代价过小则跳过
            if (*costData * 1000000.0f < 0.01f) {
                continue;
            }

            // 找到最小代价
            if (*costData < minCost) {
                if (*costData <= 0.0001f) {
                    minCost = -10.0f;
                    minDis = 1;
                } else {
                    minCost = *costData;
                    minDis = d;
                }
            }
        }

        // 将最小视差值存储到结果中
        d_rawDisp[y * width + x] = minDis;
    }
}


void DataDeal::WTAMatch(cv::Mat *&costVol, cv::Mat &disMap, int maxDis)
{
    int width = disMap.cols;
    int height = disMap.rows;

    // 分配线程块和网格
    dim3 blockSize(32, 32);  // 例如每个块16x16个线程
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 调用核函数
    wtamatchKernel<<<gridSize, blockSize>>>();

    // 检查是否有 CUDA 错误
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cout << "MatchTest over!" << endl;
}

void DataDeal::dispMapShow(std::string dispImgName, const cv::Mat &disMap)
{//��ʾ���洢�Ӳ�ͼ��
	double minVal; double maxVal;
	minMaxLoc(disMap, &minVal, &maxVal);
	Mat dispStore;
	disMap.convertTo(dispStore, CV_8UC1, 255.0 / (maxVal - minVal), -minVal*255.0 / (maxVal - minVal));//��4������ȷ�� disMap����Сֵ��Ӧ dispStore��0
	imwrite(dispImgName, dispStore);

	cout << "display disp over!" << endl;
}

void DataDeal::dispMapShowForColor(std::string dispImgName, const cv::Mat &disMap)
{//��ʾ�Ӳ�ͼ���Բ�ɫ������ʾ���洢
	double minVal; double maxVal;
	minMaxLoc(disMap, &minVal, &maxVal);
	cout << "min:" << minVal << "  max:" << maxVal;

	Mat dispStore, falseColorsMap;
	disMap.convertTo(dispStore, CV_8UC1, 255.0 / (maxVal - minVal), -minVal*255.0 / (maxVal - minVal));
	applyColorMap(dispStore, falseColorsMap, cv::COLORMAP_JET);
	imwrite(dispImgName, falseColorsMap);
	cout << "display disp over!" << endl;
}
//



/*
void DataDeal::dispMapShowForColor(std::string dispImgName, const cv::Mat &disMap) {
	// ��ʾ�Ӳ�ͼ���Բ�ɫ������ʾ���洢

	// ��ȡ�Ӳ�ͼ����С������Ӳ�ֵ
	double minVal, maxVal;
	minMaxLoc(disMap, &minVal, &maxVal);

	// �����Ӳ�ֵ������ز���
	double expandRangeStart = 12;  // Ҫ����Χ����ʼ�Ӳ�ֵ���ɵ���
	double expandRangeEnd = 13;    // Ҫ����Χ�Ľ����Ӳ�ֵ���ɵ���
	double expandTargetStart = 7;  // �����Χ����ʼ�Ӳ�ֵ���ɵ���
	double expandTargetEnd = 13;   // �����Χ����ʼ�Ӳ�ֵ���ɵ���

	cv::Mat dispStore, falseColorsMap;

	// ���Ӳ�ͼ���ݽ��д����������Ӳ�ֵ��С�����������С����
	cv::Mat processedDisMap = disMap.clone();
	for (int i = 0; i < disMap.rows; ++i) {
		for (int j = 0; j < disMap.cols; ++j) {
			// ��ȡ��ǰ���ص��Ӳ�ֵ
			double disparity = disMap.at<double>(i, j);

			// �ж��Ӳ�ֵ���ڷ�Χ��������Ӧ����
			if (disparity >= expandRangeStart && disparity <= expandRangeEnd) {
				// ���Ӳ�ֵ��Ҫ����Χ�ڽ����������
				disparity = expandTargetStart + (disparity - expandRangeStart) * (expandTargetEnd - expandTargetStart) / (expandRangeEnd - expandRangeStart);
			}
			else if (disparity < expandRangeStart) {
				// ��������Χ�������С���ֵ���ʼ�Ӳ�ֵ
				double shrinkRangeStart = minVal;
				double shrinkRangeEnd = expandRangeStart;
				double shrinkTargetStart = minVal;
				double shrinkTargetEnd = expandTargetStart;

				// ���Ӳ�ֵ��Ҫ��С��Χ�ڽ�����С����
				disparity = shrinkTargetStart + (disparity - shrinkRangeStart) * (shrinkTargetEnd - shrinkTargetStart) / (shrinkRangeEnd - shrinkRangeStart);
			}

			// ����������Ӳ�ֵ����ԭͼ��
			processedDisMap.at<double>(i, j) = disparity;
		}
	}

	// ���¼��㴦����ͼ�����Сֵ�����ֵ
	minMaxLoc(processedDisMap, &minVal, &maxVal);

	// ����������Ӳ�ͼת��Ϊ8λ�޷��ŵ�ͨ��ͼ��
	processedDisMap.convertTo(dispStore, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

	// Ӧ��ɫ��ӳ��
	applyColorMap(dispStore, falseColorsMap, cv::COLORMAP_JET);

	// ���洦����Ĳ�ɫͼ��
	imwrite(dispImgName, falseColorsMap);

	std::cout << "display disp over!" << std::endl;
}

*/

void DataDeal::readDispMapFromXML(std::string dispFileName, cv::Mat &disMap)
{//��ȡ�Ӳ�ͼ��Ϣ
	FileStorage fs;
	fs.open(dispFileName, FileStorage::READ);
	fs["imgDispMat"] >> disMap;
	fs.release();
}

void DataDeal::storeDispMapToXML(std::string dispFileName, cv::Mat &disMap)
{//�洢��Ӧ���Ӳ�״̬��Ϣ
	FileStorage fs;
	fs.open(dispFileName, FileStorage::WRITE);
	fs << "imgDispMat" << disMap;
	fs.release();
}