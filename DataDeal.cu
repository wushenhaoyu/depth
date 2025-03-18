#include "DataDeal.h"

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

__global__ void WTAMatchKernel(
    const float* costVol,
    uchar* disMap,
    int width,
    int height,
    int maxDis
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float minCost = FLT_MAX;
    int minDis = 0;

    for (int d = 1; d < maxDis; d++) {
        int costIdx = (d - 1) * width * height + y * width + x;
        float cost = costVol[costIdx];

        if (cost * 1000000.0 < 0.01)
            continue;

        if (cost < minCost) {
            if (cost <= 0.0001) {
                minCost = -10;
                minDis = 1;
            } else {
                minCost = cost;
                minDis = d;
            }
        }
    }

    int disIdx = y * width + x;
    disMap[disIdx] = minDis;
}


void DataDeal::WTAMatch(cv::Mat *&costVol, cv::Mat &disMap, int maxDis)
{//���Ӳ�dataCost�����еõ��Ӳ����ݣ���WTA�㷨
int width = disMap.cols;
    int height = disMap.rows;

    // 将 costVol 和 disMap 转换为连续的数组格式
    int costVolSize = maxDis * width * height * sizeof(float);
    int disMapSize = width * height * sizeof(uchar);

    float* d_costVol;
    uchar* d_disMap;

    // 分配设备内存
    cudaMalloc(&d_costVol, costVolSize);
    cudaMalloc(&d_disMap, disMapSize);

    // 将数据从主机内存复制到设备内存
    for (int d = 0; d < maxDis; d++) {
        cudaMemcpy(d_costVol + d * width * height, costVol[d].data, width * height * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemset(d_disMap, 0, disMapSize);

    // 计算线程块和网格大小
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    // 调用 CUDA 内核函数
    WTAMatchKernel<<<gridSize, blockSize>>>(d_costVol, d_disMap, width, height, maxDis);
    cudaDeviceSynchronize();

    // 将结果从设备内存复制回主机内存
    cudaMemcpy(disMap.data, d_disMap, disMapSize, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_costVol);
    cudaFree(d_disMap);

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