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

__global__ void wtamatchKernel(float* d_rawDisp)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int height = d_rawImageParameter.m_recImgHeight;
    int width = d_rawImageParameter.m_recImgWidth;
	//if(y > 1600)
	//printf("x:%d y:%d\n", x, y);
	
	


    if (x < width && y < height)
    {
        float minCost = d_fltMax;
        float minDis = 0;
        for (int d = 0; d < d_disparityParameter.m_disNum; d++) {
			
			int costIdx = d * width * height + y * width + x;
            float* costData = &d_costVol[costIdx];
            if (*costData * 1000000.0f < 0.01f)
                continue;

            if (*costData < minCost) {
                if (*costData <= 0.0001f) {
                    minCost = -10.0f;
                    minDis = 1;
                } else {
                    minCost = costData[0];
                    minDis = d;
                }
            }
        }
		int disIdx = y * width + x;
        d_rawDisp[disIdx] = minDis;
		//if(y == 95&& x >= 890)
		if( x >= 1700);
		//printf("x:%d y:%d d:%d value:%f,res:%f\n", x, y, minDis,d_rawDisp[disIdx]);
		
    }
}


void DataDeal::WTAMatch(int width,int height, int maxDis)
{

    dim3 blockSize(32, 32);  // 每个线程块 16x16
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
	(height + blockSize.y - 1) / blockSize.y);

    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 启动 CUDA 核函数
    cudaEventRecord(start); // 记录开始时间
    wtamatchKernel<<<gridSize, blockSize>>>(d_rawDisp);
    cudaEventRecord(stop);  // 记录结束时间

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // 计算运行时间
    float milliseconds = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 销毁 CUDA 事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "MatchTest over!" << std::endl;

	
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