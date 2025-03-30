#include "DataParameter.cuh"
using namespace cv;

using namespace std;



__constant__ RawImageParameter d_rawImageParameter;
__constant__ DisparityParameter d_disparityParameter;
__constant__ FilterParameterDevice d_filterPatameterDevice; 
__device__ MicroImageParameterDevice d_microImageParameter; 
__device__ float* d_costVol;
__device__ float* d_rawDisp;
__device__ float* d_ppLensMeanDisp;
__device__ float* d_renderCache;
__device__ float* d_inputImg;
__device__ float* d_inputImgRec;
__device__ RanderMapPatch* d_ppRanderMapPatch;
__device__ float* d_tmp;
__device__ float* d_simg;
__device__ int *d_sx_begin, *d_sy_begin, *d_sx_end, *d_sy_end;
__device__ int *d_randerMapWidth, *d_randerMapHeight;

__constant__ float d_fltMax;
__constant__ int d_meanDispLenRadius;
__constant__ int d_patchScale9;
__constant__ float d_randerScale;
__constant__ int d_destWidth;
__constant__ int d_destHeight;

__global__ void testKernel() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // 在一个特定的索引位置修改 d_ppLensMeanDisp
   // printf("m_xLensNum = %d, m_yLensNum = %d\n", d_rawImageParameter.m_xLensNum, d_rawImageParameter.m_yLensNum);
    
    
    if (x < d_rawImageParameter.m_xLensNum && y < d_rawImageParameter.m_yLensNum) {
        int index = y * d_rawImageParameter.m_xLensNum + x;
        if (index == 0) {  
			if (d_ppLensMeanDisp == nullptr) {
				printf("d_ppLensMeanDisp is NULL\n");
			}
			else{
				printf("d_ppLensMeanDisp is not NULL\n");
			}
            d_ppLensMeanDisp[index] = 0;  
        }
    }
}

__global__ void testRanderMapPatchKernel(RanderMapPatch* d_ppRanderMapPatch, int numPatches, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPatches) return;

    // 获取当前 RanderMapPatch 的 simg 指针
    float* simg = d_ppRanderMapPatch[idx].simg;

    // 对 simg 指向的内存进行简单的写操作
    for (int i = 0; i < width * height * 3; ++i)
    {
        simg[i] = static_cast<float>(idx * (i + 1)); // 写入一些模式数据
    }

    // 设置 sy 和 sx 的值
    d_ppRanderMapPatch[idx].sy = idx * 10; // 示例值
    d_ppRanderMapPatch[idx].sx = idx * 20; // 示例值


    printf("Patch 0: sy = %d, sx = %d, simg[0] = %f, simg[1] = %f, simg[2] = %f\n",
               d_ppRanderMapPatch[idx].sy, d_ppRanderMapPatch[idx].sx,
               simg[0], simg[1], simg[2]);
    
}

__global__ void testAssignAndPrintKernel()
{
    // 每个线程对全局变量赋值
    *d_sx_begin = 10;
    *d_sy_begin = 20;
    *d_sx_end = 30;
    *d_sy_end = 40;

    // 打印全局变量的值
    printf("d_sx_begin: %d\n", *d_sx_begin);
    printf("d_sy_begin: %d\n", *d_sy_begin);
    printf("d_sx_end: %d\n", *d_sx_end);
    printf("d_sy_end: %d\n", *d_sy_end);
}

void DataParameter::mapToGPU() {
    // 复制常量变量
    CUDA_CHECK(cudaMemcpyToSymbol(d_rawImageParameter, &m_rawImageParameter, sizeof(RawImageParameter)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_disparityParameter, &m_disparityParameter, sizeof(DisparityParameter)));

    float fltMax = FLT_MAX;
    CUDA_CHECK(cudaMemcpyToSymbol(d_fltMax, &fltMax, sizeof(float)));

    int meanDispLenRadius = MEAN_DISP_LEN_RADIUS;
    int patchScale9 = PATCH_SCALE9;
    float randerScale = RANDER_SCALE;
    int destWidth = DEST_WIDTH;
    int destHeight = DEST_HEIGHT;

    CUDA_CHECK(cudaMemcpyToSymbol(d_meanDispLenRadius, &meanDispLenRadius, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_patchScale9, &patchScale9, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_randerScale, &randerScale, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_destWidth, &destWidth, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_destHeight, &destHeight, sizeof(int)));

    // 在主机端分配内存并将指针传递给设备端
    float* h_costVol;
    CUDA_CHECK(cudaMalloc((void**)&h_costVol, m_disparityParameter.m_disNum * m_rawImageParameter.m_recImgHeight * m_rawImageParameter.m_recImgWidth * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_costVol, &h_costVol, sizeof(float*)));

    float* h_rawDisp;
    CUDA_CHECK(cudaMalloc((void**)&h_rawDisp, m_rawImageParameter.m_recImgHeight * m_rawImageParameter.m_recImgWidth * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_rawDisp, &h_rawDisp, sizeof(float*)));
	

    float* h_ppLensMeanDisp;
    CUDA_CHECK(cudaMalloc((void**)&h_ppLensMeanDisp, m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_ppLensMeanDisp, &h_ppLensMeanDisp, sizeof(float*)));

    float* h_renderCache;
    CUDA_CHECK(cudaMalloc((void**)&h_renderCache, m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_renderCache, &h_renderCache, sizeof(float*)));

    /*float* h_inputImg;
    CUDA_CHECK(cudaMalloc((void**)&h_inputImg, m_rawImageParameter.m_srcImgHeight * m_rawImageParameter.m_srcImgWidth * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_inputImg, &h_inputImg, sizeof(float*)));

    float* h_inputImgRec;
    CUDA_CHECK(cudaMalloc((void**)&h_inputImgRec, m_rawImageParameter.m_recImgHeight * m_rawImageParameter.m_recImgWidth * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_inputImgRec, &h_inputImgRec, sizeof(float*)));*/


int height = m_rawImageParameter.m_recImgHeight;
int width = m_rawImageParameter.m_recImgWidth;
int sizeOfFloat = (int)sizeof(float);

    float* h_tmp;
    CUDA_CHECK(cudaMalloc((void**)&h_tmp, DEST_WIDTH * DEST_HEIGHT * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_tmp, &h_tmp, sizeof(float*)));

    float* h_simg;
    CUDA_CHECK(cudaMalloc((void**)&h_simg, DEST_WIDTH * DEST_HEIGHT * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_simg, &h_simg, sizeof(float*)));

	int* h_sx_begin;
	CUDA_CHECK(cudaMalloc((void**)&h_sx_begin, sizeof(int)));
	CUDA_CHECK(cudaMemcpyToSymbol(d_sx_begin, &h_sx_begin, sizeof(int*)));


	int* h_sy_begin;
	CUDA_CHECK(cudaMalloc((void**)&h_sy_begin, sizeof(int)));
	CUDA_CHECK(cudaMemcpyToSymbol(d_sy_begin, &h_sy_begin, sizeof(int*)));

	int* h_sx_end;
	CUDA_CHECK(cudaMalloc((void**)&h_sx_end, sizeof(int)));
	CUDA_CHECK(cudaMemcpyToSymbol(d_sx_end, &h_sx_end, sizeof(int*)));

	int* h_sy_end;
	CUDA_CHECK(cudaMalloc((void**)&h_sy_end, sizeof(int)));
	CUDA_CHECK(cudaMemcpyToSymbol(d_sy_end, &h_sy_end, sizeof(int*)));

	int* h_randerMapWidth;
    CUDA_CHECK(cudaMalloc((void**)&h_randerMapWidth, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_randerMapWidth, &h_randerMapWidth, sizeof(int*)));

	int* h_randerMapHeight;
    CUDA_CHECK(cudaMalloc((void**)&h_randerMapHeight, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_randerMapHeight, &h_randerMapHeight, sizeof(int*)));

    // 为 MicroImageParameterDevice 分配内存并传递数据到设备
    MicroImageParameterDevice h_microImageParameterDevice;
    h_microImageParameterDevice.m_circleDiameter = m_microImageParameter.m_circleDiameter;
    h_microImageParameterDevice.m_circleNarrow = m_microImageParameter.m_circleNarrow;
    h_microImageParameterDevice.m_radiusDisEqu = m_microImageParameter.m_radiusDisEqu;

    int lensCenterPointsSize = m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum * sizeof(cv::Point2d);
    CUDA_CHECK(cudaMalloc((void**)&h_microImageParameterDevice.m_ppLensCenterPoints, lensCenterPointsSize));
    CUDA_CHECK(cudaMemcpy(h_microImageParameterDevice.m_ppLensCenterPoints, m_microImageParameter.m_ppLensCenterPoints[0], lensCenterPointsSize, cudaMemcpyHostToDevice));

    int pixelsMappingSetSize = m_rawImageParameter.m_srcImgHeight * m_rawImageParameter.m_srcImgWidth * sizeof(int);
    CUDA_CHECK(cudaMalloc((void**)&h_microImageParameterDevice.m_ppPixelsMappingSet, pixelsMappingSetSize));
    CUDA_CHECK(cudaMemcpy(h_microImageParameterDevice.m_ppPixelsMappingSet, m_microImageParameter.m_ppPixelsMappingSet[0], pixelsMappingSetSize, cudaMemcpyHostToDevice));

    int matchNeighborLensSize = m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum * NEIGHBOR_MATCH_LENS_NUM * sizeof(MatchNeighborLens);
    CUDA_CHECK(cudaMalloc((void**)&h_microImageParameterDevice.m_ppMatchNeighborLens, matchNeighborLensSize));
    CUDA_CHECK(cudaMemcpy(h_microImageParameterDevice.m_ppMatchNeighborLens, m_microImageParameter.m_ppMatchNeighborLens[0][0], matchNeighborLensSize, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(d_microImageParameter, &h_microImageParameterDevice, sizeof(MicroImageParameterDevice)));

    // 为 FilterParameterDevice 分配内存并传递数据到设备
    int* d_validNeighborPixelsNum;
    int* d_validPixelsMask;
    float* d_filterKernel;

    CUDA_CHECK(cudaMalloc((void**)&d_validNeighborPixelsNum, m_filterPatameter.m_pValidNeighborPixelsNum->total() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_validNeighborPixelsNum, m_filterPatameter.m_pValidNeighborPixelsNum->data, 
                          m_filterPatameter.m_pValidNeighborPixelsNum->total() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_validPixelsMask, m_filterPatameter.m_pValidPixelsMask->total() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_validPixelsMask, m_filterPatameter.m_pValidPixelsMask->data, 
                          m_filterPatameter.m_pValidPixelsMask->total() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_filterKernel, m_filterPatameter.m_filterKnernel.total() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_filterKernel, m_filterPatameter.m_filterKnernel.data, 
                          m_filterPatameter.m_filterKnernel.total() * sizeof(float), cudaMemcpyHostToDevice));

    FilterParameterDevice filterParamDevice = { d_validNeighborPixelsNum, d_validPixelsMask, d_filterKernel };
    CUDA_CHECK(cudaMemcpyToSymbol(d_filterPatameterDevice, &filterParamDevice, sizeof(FilterParameterDevice)));
	
	    // 1. 主机端分配 RanderMapPatch 数组
		RanderMapPatch* h_ppRanderMapPatch = new RanderMapPatch[m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum];

		// 2. 为每个 RanderMapPatch 的 simg 分配设备内存
		float** h_simgDevice = new float*[m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum];
		for (int i = 0; i < m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum; ++i) {
			CUDA_CHECK(cudaMalloc((void**)&h_simgDevice[i], DEST_WIDTH * DEST_HEIGHT * 3 * sizeof(float)));
			h_ppRanderMapPatch[i].simg = h_simgDevice[i]; // 将设备内存地址存储在主机端的 RanderMapPatch 中
		}
	
		// 3. 设备端分配 RanderMapPatch 数组
		CUDA_CHECK(cudaMalloc((void**)&d_ppRanderMapPatch, m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum * sizeof(RanderMapPatch)));
	
		// 4. 复制结构体数组到设备端
		CUDA_CHECK(cudaMemcpy(d_ppRanderMapPatch, h_ppRanderMapPatch, m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum * sizeof(RanderMapPatch), cudaMemcpyHostToDevice));
	
	
		// 6. 释放主机端数据
		delete[] h_ppRanderMapPatch;
		delete[] h_simgDevice;
	
}

void DataParameter::UpdateImgToGPU()
{
    // 确保输入图像和裁剪后的图像是连续的
    cv::Mat inputImgContinuous = m_inputImg.isContinuous() ? m_inputImg : m_inputImg.clone();
    cv::Mat inputImgRecContinuous = m_inputImgRec.isContinuous() ? m_inputImgRec : m_inputImgRec.clone();

    // 计算图像的字节大小
    size_t inputImgSize = inputImgContinuous.total() * inputImgContinuous.elemSize();
    size_t inputImgRecSize = inputImgRecContinuous.total() * inputImgRecContinuous.elemSize();

    // 为输入图像分配内存
	CUDA_CHECK(cudaMalloc((void**)&d_inputImg, inputImgSize));

	// 为裁剪后的图像分配内存
	CUDA_CHECK(cudaMalloc((void**)&d_inputImgRec, inputImgRecSize));

    // 将输入图像数据从主机复制到设备
    CUDA_CHECK(cudaMemcpy(d_inputImg, inputImgContinuous.ptr<float>(0), inputImgSize, cudaMemcpyHostToDevice));

    // 将裁剪后的图像数据从主机复制到设备
    CUDA_CHECK(cudaMemcpy(d_inputImgRec, inputImgRecContinuous.ptr<float>(0), inputImgRecSize, cudaMemcpyHostToDevice));

    // 将设备指针复制到 __device__ 变量
    CUDA_CHECK(cudaMemcpyToSymbol(d_inputImg, &d_inputImg, sizeof(float*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_inputImgRec, &d_inputImgRec, sizeof(float*)));
}
	


DataParameter::DataParameter()
{
	m_rawImageParameter.m_yCenterBeginOffset = 2;
	m_rawImageParameter.m_xCenterBeginOffset = 2;
	m_rawImageParameter.m_yCenterEndOffset = 2;
	m_rawImageParameter.m_xCenterEndOffset = 2;
	m_filterRadius = 4;
	m_microImageParameter.m_circleDiameter = 34.0;
	m_microImageParameter.m_circleNarrow = 1.5;
	m_disparityParameter.m_dispMin = 5;
	m_disparityParameter.m_dispMax = 13;
	m_disparityParameter.m_dispStep = 0.5;
	m_microImageParameter.m_ppLensCenterPoints = nullptr;
	m_microImageParameter.m_ppPixelsMappingSet = nullptr;
	m_microImageParameter.m_ppMatchNeighborLens = nullptr;
	m_filterPatameter.m_pValidNeighborPixelsNum = nullptr;
	m_filterPatameter.m_pValidPixelsMask = nullptr;
}

DataParameter::DataParameter(std::string dataFolderName, std::string centerPointFileName, std::string inputImgName,
	int yCenterBeginOffset, int xCenterBeginOffset, int yCenterEndOffset, int xCenterEndOffset,
	int filterRadius, float circleDiameter, float circleNarrow, int dispMin, int dispMax, float dispStep)
{
	init(dataFolderName, centerPointFileName, inputImgName, yCenterBeginOffset, xCenterBeginOffset, yCenterEndOffset, xCenterEndOffset,
		filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);
}

DataParameter::~DataParameter()//�����������������ڽ������Զ�����
{
	if (m_microImageParameter.m_ppLensCenterPoints)
	{
		for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
			delete[]m_microImageParameter.m_ppLensCenterPoints[y];
		delete[]m_microImageParameter.m_ppLensCenterPoints;
	}
	if (m_microImageParameter.m_ppPixelsMappingSet)
	{
		for (int y = 0; y < m_rawImageParameter.m_srcImgHeight; y++)
			delete[]m_microImageParameter.m_ppPixelsMappingSet[y];
		delete[]m_microImageParameter.m_ppPixelsMappingSet;
	}
	if (m_microImageParameter.m_ppMatchNeighborLens)
	{
		for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
			delete[]m_microImageParameter.m_ppMatchNeighborLens[y];
		delete[]m_microImageParameter.m_ppMatchNeighborLens;
	}
	if (m_filterPatameter.m_pValidPixelsMask)
	{
		delete m_filterPatameter.m_pValidPixelsMask;
	}
	if (m_filterPatameter.m_pValidNeighborPixelsNum)
	{
		delete m_filterPatameter.m_pValidNeighborPixelsNum;
	}
}

void DataParameter::init(std::string dataFolderName, std::string centerPointFileName, std::string inputImgName,
	int yCenterBeginOffset, int xCenterBeginOffset, int yCenterEndOffset, int xCenterEndOffset,
	int filterRadius, float circleDiameter, float circleNarrow, int dispMin, int dispMax, float dispStep)
{
	lensCenterPointsInit(dataFolderName, centerPointFileName);
	validLensCenterInit(yCenterBeginOffset, xCenterBeginOffset, yCenterEndOffset, xCenterEndOffset);
	imageBaseMessageInit(inputImgName, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);
	generatePixelsMappingSet();
	generateNeighborCenterPoints();
	generateValidPoints();
	mapToGPU();

	std::cout << "DataParameter::init final!" << std::endl;
}

void DataParameter::lensCenterPointsInit(std::string dataFolderPath, std::string centerPointFileName)
{//͸�����ĵ��ʼ��
	m_folderPath = dataFolderPath;
	std::string centerFileTxtName = dataFolderPath + "/" + centerPointFileName;
	std::ifstream ifs;
	ifs.open(centerFileTxtName, std::ifstream::in);
	ifs >> m_rawImageParameter.m_yLensNum >> m_rawImageParameter.m_xLensNum;//���ļ��ж�ȡ͸�������� m_yLensNum ������ m_xLensNum�����洢�� m_rawImageParameter �ṹ���С�
	m_microImageParameter.m_ppLensCenterPoints = new Point2d *[m_rawImageParameter.m_yLensNum];
	for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
		m_microImageParameter.m_ppLensCenterPoints[y] = new Point2d[m_rawImageParameter.m_xLensNum];

	for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
		for (int x = 0; x < m_rawImageParameter.m_xLensNum; x++)
			ifs >> m_microImageParameter.m_ppLensCenterPoints[y][x].y >> m_microImageParameter.m_ppLensCenterPoints[y][x].x;

	ifs.close();
	float x1 = m_microImageParameter.m_ppLensCenterPoints[0][0].x; //��0����ߵ�һ��x����
	float x2 = m_microImageParameter.m_ppLensCenterPoints[1][0].x; //��1����ߵ�һ��x����

	if (x1 > x2) m_lensArrageMode = 0; //��0�е�һ��Բ������ 
	else m_lensArrageMode = 1; //��0�е�һ��Բ����͹��
	std::cout << "lensCenterPointsInit final!" << std::endl;
}

void DataParameter::validLensCenterInit(int yCenterBeginOffset, int xCenterBeginOffset, int yCenterEndOffset, int xCenterEndOffset)
{//�����Щ͸��������Ч
	m_rawImageParameter.m_yCenterBeginOffset = yCenterBeginOffset;
	m_rawImageParameter.m_xCenterBeginOffset = xCenterBeginOffset;
	m_rawImageParameter.m_yCenterEndOffset = yCenterEndOffset;
	m_rawImageParameter.m_xCenterEndOffset = xCenterEndOffset;

	Point2d &topLeftCenterPos = m_microImageParameter.m_ppLensCenterPoints[yCenterBeginOffset - 1][xCenterBeginOffset - 1];
	Point2d &topRightCenterPos = m_microImageParameter.m_ppLensCenterPoints[yCenterBeginOffset - 1][m_rawImageParameter.m_xLensNum - xCenterEndOffset];
	Point2d &belowLeftCenterPos = m_microImageParameter.m_ppLensCenterPoints[m_rawImageParameter.m_yLensNum - yCenterEndOffset][xCenterBeginOffset - 1];
	Point2d &belowRightCenterPos = m_microImageParameter.m_ppLensCenterPoints[m_rawImageParameter.m_yLensNum - yCenterEndOffset][m_rawImageParameter.m_xLensNum - xCenterEndOffset];

	double left = std::min(topLeftCenterPos.x, belowLeftCenterPos.x);
	double right = std::max(topRightCenterPos.x, belowRightCenterPos.x);
	double top = std::min(topLeftCenterPos.y, topRightCenterPos.y);
	double below = std::max(belowLeftCenterPos.y, belowRightCenterPos.y);

	m_rawImageParameter.m_yPixelBeginOffset = top;
	m_rawImageParameter.m_xPixelBeginOffset = left;

	m_rawImageParameter.m_recImgHeight = below - top + 1;
	m_rawImageParameter.m_recImgWidth = right - left + 1;
	cout<<"validLensCenterInit_final"<<endl;
}

void DataParameter::imageBaseMessageInit(std::string inputImgName, int filterRadius, float circleDiameter,
	float circleNarrow, int dispMin, int dispMax, float dispStep)
{//ͼ�������Ϣ���Ӳ���Ϣ��ʼ��
	cout<<"imageBaseMessageInit_start"<<endl;
	m_filterRadius = filterRadius;//�˲��뾶
	m_microImageParameter.m_circleDiameter = circleDiameter;//԰ֱ��
	m_microImageParameter.m_circleNarrow = circleNarrow;//Բ�뾶��Сֵ
	m_disparityParameter.m_dispMin = dispMin;
	m_disparityParameter.m_dispMax = dispMax;
	m_disparityParameter.m_dispStep = dispStep;
	m_disparityParameter.m_disNum = double(dispMax - dispMin) / m_disparityParameter.m_dispStep; //�Ӳ�label��Ŀ
	m_filterPatameter.m_filterKnernel = cv::Mat::ones(2 * filterRadius + 1, 2 * filterRadius + 1, CV_32FC1);//�����˲��뾶���������
	m_microImageParameter.m_radiusDisEqu = (circleDiameter / 2 - m_microImageParameter.m_circleNarrow)*(circleDiameter / 2 - m_microImageParameter.m_circleNarrow);
	std::string inputImagePath = m_folderPath + "/" + inputImgName;
	cout<<"m_folderPath："<<m_folderPath<<endl;
	cout<<"inputImagePath："<<inputImagePath<<endl;
	
	m_inputImg = imread(inputImagePath, IMREAD_COLOR);
	
	//m_inputImg = imread(inputImagePath, IMREAD_GRAYSCALE);
	m_rawImageParameter.m_srcImgWidth = m_inputImg.cols;
	m_rawImageParameter.m_srcImgHeight = m_inputImg.rows;


	m_inputImgRec = m_inputImg(cv::Rect(m_rawImageParameter.m_xPixelBeginOffset, m_rawImageParameter.m_yPixelBeginOffset, 
		m_rawImageParameter.m_recImgWidth, m_rawImageParameter.m_recImgHeight)).clone();

	UpdateImgToGPU();

	std::string recImageStore = m_folderPath + "/" + "srcImgRec.png";
	imwrite(recImageStore, m_inputImgRec);
	cout<<"imageBaseMessageInit_final"<<endl;
}

void DataParameter::generatePixelsMappingSet()
{//�������ص�ӳ�伯��
	m_microImageParameter.m_ppPixelsMappingSet = new int *[m_rawImageParameter.m_srcImgHeight];
	for (int y = 0; y < m_rawImageParameter.m_srcImgHeight; y++)
	{
		m_microImageParameter.m_ppPixelsMappingSet[y] = new int[m_rawImageParameter.m_srcImgWidth];
		memset(m_microImageParameter.m_ppPixelsMappingSet[y], -1, m_rawImageParameter.m_srcImgWidth*sizeof(int));
	}

	//�����Щ��������Ч��
	m_filterPatameter.m_pValidPixelsMask = new cv::Mat;
	*m_filterPatameter.m_pValidPixelsMask = Mat::zeros(m_rawImageParameter.m_srcImgHeight, m_rawImageParameter.m_srcImgWidth, CV_32FC1);
//#pragma omp parallel for 
	for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
		for (int x = 0; x < m_rawImageParameter.m_xLensNum; x++)
			generatePixelsMappingSet(y, x);

	std::cout << "generatePixelsMappingSet final!" << std::endl;
}

void DataParameter::generatePixelsMappingSet(int y, int x)
{//�������ص�ӳ�伯��--����ÿ����ͼ���cost
	Point2d &centerPos = m_microImageParameter.m_ppLensCenterPoints[y][x];
	//������ǰ͸������λ�ø���������
	for (int py = centerPos.y - m_microImageParameter.m_circleDiameter / 2 + m_microImageParameter.m_circleNarrow; 
		py <= centerPos.y + m_microImageParameter.m_circleDiameter / 2 - m_microImageParameter.m_circleNarrow; py++)
	{
		//�����������Ϊ1������ô�����������Ч
		float *yDataRowsMask = (float *)(*m_filterPatameter.m_pValidPixelsMask).ptr<float *>(py);
		for (int px = centerPos.x - m_microImageParameter.m_circleDiameter / 2 + m_microImageParameter.m_circleNarrow; 
			px <= centerPos.x + m_microImageParameter.m_circleDiameter / 2 - m_microImageParameter.m_circleNarrow; px++)
		{
			//΢͸��������
			int lens_num = y*m_rawImageParameter.m_xLensNum + x;
			//��������Ƿ���Բ��
			if ((centerPos.y - py)*(centerPos.y - py) + (centerPos.x - px)*(centerPos.x - px) <= m_microImageParameter.m_radiusDisEqu)
			{
				m_microImageParameter.m_ppPixelsMappingSet[py][px] = lens_num;
				yDataRowsMask[px] = 1.0;
			}
		}
	}
}

void DataParameter::generateNeighborCenterPoints()
{//����ÿ�����ĵ���Χ�Ķ�Ӧ���ĵ�
	//ÿ��͸�����ĵ���ھ����ĵ���Ϣ
	m_microImageParameter.m_ppMatchNeighborLens = new MatchNeighborLens **[m_rawImageParameter.m_yLensNum];
	for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
	{
		m_microImageParameter.m_ppMatchNeighborLens[y] = new MatchNeighborLens *[m_rawImageParameter.m_xLensNum];
		for (int x = 0; x < m_rawImageParameter.m_xLensNum; x++)
			m_microImageParameter.m_ppMatchNeighborLens[y][x] = new MatchNeighborLens[NEIGHBOR_MATCH_LENS_NUM];
	}

//#pragma omp parallel for 
	for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
		for (int x = 0; x < m_rawImageParameter.m_xLensNum; x++)
			generateNeighborCenterPoints(y, x);

	std::cout << "generateNeighborCenterPoints final!" << std::endl;
}

void DataParameter::generateNeighborCenterPoints(int y, int x)
{//����ÿ�����ĵ���Χ�Ķ�Ӧ���ĵ�
	int parityFlag = y & 1;//������Ϊ1��ż����Ϊ0
	int x_shift = 0;
	if (m_lensArrageMode ^ parityFlag){
		//��͹��ż���У������󰼣�������
		x_shift = -1;
	}
	else{
		//��͹�������У������󰼣�ż����
		x_shift = 1;
	}

	int numCount = 0;
	Point2d &curCenterPoint = m_microImageParameter.m_ppLensCenterPoints[y][x];
	double cy, cx, dis;

	
	if (y - 1 >= 0){//�Ϸ��ھӣ��൱���������������������������
		cy = m_microImageParameter.m_ppLensCenterPoints[y - 1][x].y;
		cx = m_microImageParameter.m_ppLensCenterPoints[y - 1][x].x;
		dis = sqrt((cy - curCenterPoint.y)*(cy - curCenterPoint.y) + (cx - curCenterPoint.x)*(cx - curCenterPoint.x));
		m_microImageParameter.m_ppMatchNeighborLens[y][x][numCount++] = MatchNeighborLens(cy, cx, dis, (y - 1)*m_rawImageParameter.m_xLensNum + x);

		if ((x_shift < 0 && x - 1 >= 0) || (x_shift > 0 && x + 1 < m_rawImageParameter.m_xLensNum)){
			cy = m_microImageParameter.m_ppLensCenterPoints[y - 1][x + x_shift].y;
			cx = m_microImageParameter.m_ppLensCenterPoints[y - 1][x + x_shift].x;
			dis = sqrt((cy - curCenterPoint.y)*(cy - curCenterPoint.y) + (cx - curCenterPoint.x)*(cx - curCenterPoint.x));
			m_microImageParameter.m_ppMatchNeighborLens[y][x][numCount++] = MatchNeighborLens(cy, cx, dis, (y - 1)*m_rawImageParameter.m_xLensNum + x + x_shift);
		}
	}

	if (y + 1 < m_rawImageParameter.m_yLensNum){//�·��ھ�
		cy = m_microImageParameter.m_ppLensCenterPoints[y + 1][x].y;
		cx = m_microImageParameter.m_ppLensCenterPoints[y + 1][x].x;
		dis = sqrt((cy - curCenterPoint.y)*(cy - curCenterPoint.y) + (cx - curCenterPoint.x)*(cx - curCenterPoint.x));
		m_microImageParameter.m_ppMatchNeighborLens[y][x][numCount++] = MatchNeighborLens(cy, cx, dis, (y + 1)*m_rawImageParameter.m_xLensNum + x);

		if ((x_shift < 0 && x - 1 >= 0) || (x_shift > 0 && x + 1 < m_rawImageParameter.m_xLensNum)){
			cy = m_microImageParameter.m_ppLensCenterPoints[y + 1][x + x_shift].y;
			cx = m_microImageParameter.m_ppLensCenterPoints[y + 1][x + x_shift].x;
			dis = sqrt((cy - curCenterPoint.y)*(cy - curCenterPoint.y) + (cx - curCenterPoint.x)*(cx - curCenterPoint.x));
			m_microImageParameter.m_ppMatchNeighborLens[y][x][numCount++] = MatchNeighborLens(cy, cx, dis, (y + 1)*m_rawImageParameter.m_xLensNum + x + x_shift);
		}
	}

	if (x - 1 >= 0){//����ھ�
		cy = m_microImageParameter.m_ppLensCenterPoints[y][x - 1].y;
		cx = m_microImageParameter.m_ppLensCenterPoints[y][x - 1].x;
		dis = sqrt((cy - curCenterPoint.y)*(cy - curCenterPoint.y) + (cx - curCenterPoint.x)*(cx - curCenterPoint.x));
		m_microImageParameter.m_ppMatchNeighborLens[y][x][numCount++] = MatchNeighborLens(cy, cx, dis, y*m_rawImageParameter.m_xLensNum + x - 1);
	}

	if (x + 1 < m_rawImageParameter.m_xLensNum){//�Ҳ��ھ�
		cy = m_microImageParameter.m_ppLensCenterPoints[y][x + 1].y;
		cx = m_microImageParameter.m_ppLensCenterPoints[y][x + 1].x;
		dis = sqrt((cy - curCenterPoint.y)*(cy - curCenterPoint.y) + (cx - curCenterPoint.x)*(cx - curCenterPoint.x));
		m_microImageParameter.m_ppMatchNeighborLens[y][x][numCount++] = MatchNeighborLens(cy, cx, dis, y*m_rawImageParameter.m_xLensNum + x + 1);
	}

	if (numCount < NEIGHBOR_MATCH_LENS_NUM)//�����������Ϣ
		m_microImageParameter.m_ppMatchNeighborLens[y][x][numCount++] = MatchNeighborLens(-1, -1, -1, -1);
	
}

void DataParameter::generateValidPoints()
{//����ÿ��СԲ�����أ�����Χ��Ч�����
	m_filterPatameter.m_pValidNeighborPixelsNum = new cv::Mat;
	*m_filterPatameter.m_pValidNeighborPixelsNum = cv::Mat::zeros(m_rawImageParameter.m_srcImgHeight, m_rawImageParameter.m_srcImgWidth, CV_32FC1);

//#pragma omp parallel for
	for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
		for (int x = 0; x < m_rawImageParameter.m_xLensNum; x++)
			generateValidPoints(y, x);

	validPointsBoundaryRepair();
	std::cout << "generateValidPoints final!" << std::endl;
}

void DataParameter::generateValidPoints(int y, int x)
{//����ÿ��СԲ�����أ�����Χ��Ч�����
	Point2d &curCenterPos = m_microImageParameter.m_ppLensCenterPoints[y][x];
	int x_begin = curCenterPos.x - m_microImageParameter.m_circleDiameter / 2 + m_microImageParameter.m_circleNarrow;
	int y_begin = curCenterPos.y - m_microImageParameter.m_circleDiameter / 2 + m_microImageParameter.m_circleNarrow;
	int x_end = curCenterPos.x + m_microImageParameter.m_circleDiameter / 2 - m_microImageParameter.m_circleNarrow;
	int y_end = curCenterPos.y + m_microImageParameter.m_circleDiameter / 2 - m_microImageParameter.m_circleNarrow;

	//��ȡСԲ����
	cv::Mat srcCost = (*m_filterPatameter.m_pValidPixelsMask)(cv::Rect(x_begin, y_begin, x_end - x_begin + 1, y_end - y_begin + 1));
	cv::Mat destCost = (*m_filterPatameter.m_pValidNeighborPixelsNum)(cv::Rect(x_begin, y_begin, x_end - x_begin + 1, y_end - y_begin + 1));



	cv::filter2D(srcCost, destCost, -1, m_filterPatameter.m_filterKnernel, cv::Point(-1, -1), 0, BORDER_CONSTANT);
	destCost = destCost.mul(srcCost);//������ЧԪ�أ�ͳ����Ч�ھ���
}

void DataParameter::validPointsBoundaryRepair()
{//����Ч��ı߽��������ؽ�����1��������ֹ���ֳ�0����
//#pragma omp parallel for
	for (int py = 0; py < m_rawImageParameter.m_srcImgHeight; py++)
	{
		float *yDataRows = (float *)(*m_filterPatameter.m_pValidNeighborPixelsNum).ptr<float *>(py);
		for (int px = 0; px < m_rawImageParameter.m_srcImgWidth; px++)
		{
			if (yDataRows[px] < 0.1) // �������Ч�����ٻ� >=1
				yDataRows[px] = 1.0;
		}
	}
}