#include "DataParameter.cuh"
using namespace cv;

using namespace std;


/*没有经验,蠢猪写法*/
__constant__ RawImageParameter d_rawImageParameter;
__constant__ DisparityParameter d_disparityParameter;
FilterParameterDevice* d_filterPatameterDevice; 
MicroImageParameterDevice* d_microImageParameter;
__device__ float* d_costVol;
 float* d_rawDisp;
 float* d_rawDisp_temp;
__device__ float* d_ppLensMeanDisp;
 float* d_inputImg;
 float* d_inputImgRec;
 float* d_grayImg;
 float* d_gradImg;
RanderMapPatch* d_ppRanderMapPatch;
int* d_sx_begin;
int* d_sy_begin;
int* d_sx_end;
int* d_sy_end;
__device__ int *d_randerMapWidth, *d_randerMapHeight;
__constant__ float d_fltMax;
__constant__ float d_filterRadius;
__constant__ int d_meanDispLenRadius;
__constant__ int d_patchScale9;
__constant__ float d_randerScale;
__constant__ int d_destWidth;
__constant__ int d_destHeight;




__global__ void zeroFillSimg(RanderMapPatch* d_ppRanderMapPatch, int numPatches, int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPatches) {
		//printf("idx = %d\n", idx); // 打印当前索引
        RanderMapPatch patch = d_ppRanderMapPatch[idx];
        float* simg = patch.simg;
        int totalPixels = width * height * channels;
        for (int i = 0; i < totalPixels; ++i) {
            simg[i] = 0.0f;
			//printf("simg[%d] = %f\n", i, simg[i]); // 打印每个像素的值
        }
    }
}

__global__ void printLensCenterPoints_1(MicroImageParameterDevice* d_microImageParameterDevice, int yLensNum, int xLensNum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < yLensNum * xLensNum) {
        int y = idx / xLensNum;
        int x = idx % xLensNum;

        // 获取当前点的坐标
        CudaPoint2f point = d_microImageParameterDevice->m_ppLensCenterPoints[idx];
        //printf("x:%d,y:%d,px:%d,py:%d\n",x,y,point.x,point.y);

    }
}


void DataParameter::mapToGPU() {
    // 复制常量变量
	// 打印 CPU 端的 rawImageParameter 数据

	// 将数据复制到 GPU
	CUDA_CHECK(cudaMemcpyToSymbol(d_rawImageParameter, &m_rawImageParameter, sizeof(RawImageParameter)));

	// 从 GPU 端读取数据
	RawImageParameter h_rawImageParameterGPU;
	CUDA_CHECK(cudaMemcpyFromSymbol(&h_rawImageParameterGPU, d_rawImageParameter, sizeof(RawImageParameter)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_disparityParameter, &m_disparityParameter, sizeof(DisparityParameter)));

    float fltMax = FLT_MAX;
    CUDA_CHECK(cudaMemcpyToSymbol(d_fltMax, &fltMax, sizeof(float)));

	float filterRadius = 6;
	CUDA_CHECK(cudaMemcpyToSymbol(d_filterRadius, &filterRadius, sizeof(float)));

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


	CUDA_CHECK(cudaMalloc((void**)&d_rawDisp, m_rawImageParameter.m_recImgHeight * m_rawImageParameter.m_recImgWidth * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_rawDisp, 0, m_rawImageParameter.m_recImgHeight * m_rawImageParameter.m_recImgWidth * sizeof(float)));

	
	CUDA_CHECK(cudaMalloc((void**)&d_rawDisp_temp, m_rawImageParameter.m_recImgHeight * m_rawImageParameter.m_recImgWidth * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_rawDisp_temp, 0, m_rawImageParameter.m_recImgHeight * m_rawImageParameter.m_recImgWidth * sizeof(float)));


    float* h_ppLensMeanDisp;
    CUDA_CHECK(cudaMalloc((void**)&h_ppLensMeanDisp, m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_ppLensMeanDisp, &h_ppLensMeanDisp, sizeof(float*)));


	CUDA_CHECK(cudaMalloc((void**)&d_grayImg, m_rawImageParameter.m_srcImgHeight * m_rawImageParameter.m_srcImgWidth * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grayImg, 0,m_rawImageParameter.m_srcImgHeight * m_rawImageParameter.m_srcImgWidth * sizeof(float)));

	
	CUDA_CHECK(cudaMalloc((void**)&d_gradImg, m_rawImageParameter.m_srcImgHeight * m_rawImageParameter.m_srcImgWidth * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gradImg, 0,m_rawImageParameter.m_srcImgHeight * m_rawImageParameter.m_srcImgWidth * sizeof(float)));


	int height = m_rawImageParameter.m_recImgHeight;
	int width = m_rawImageParameter.m_recImgWidth;
	int sizeOfFloat = (int)sizeof(float);

	int h_sx_begin_val = INT_MAX;
	int h_sy_begin_val = INT_MAX;
	int h_sx_end_val = INT_MIN;
	int h_sy_end_val = INT_MIN;
	CUDA_CHECK(cudaMalloc((void**)&d_sx_begin, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_sy_begin, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_sx_end, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_sy_end, sizeof(int)));

    // 初始化设备内存
    CUDA_CHECK(cudaMemcpy(d_sx_begin, &h_sx_begin_val, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sy_begin, &h_sy_begin_val, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sx_end, &h_sx_end_val, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sy_end, &h_sy_end_val, sizeof(int), cudaMemcpyHostToDevice));

	int* h_randerMapWidth;
    CUDA_CHECK(cudaMalloc((void**)&h_randerMapWidth, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_randerMapWidth, &h_randerMapWidth, sizeof(int*)));

	int* h_randerMapHeight;
    CUDA_CHECK(cudaMalloc((void**)&h_randerMapHeight, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_randerMapHeight, &h_randerMapHeight, sizeof(int*)));

    // 创建 MicroImageParameterDevice 实例
    MicroImageParameterDevice h_microImageParameterDevice;
    h_microImageParameterDevice.m_circleDiameter = m_microImageParameter.m_circleDiameter;
    h_microImageParameterDevice.m_circleNarrow = m_microImageParameter.m_circleNarrow;
    h_microImageParameterDevice.m_radiusDisEqu = m_microImageParameter.m_radiusDisEqu;

    // 为 m_ppLensCenterPoints 分配内存并复制数据
    int lensCenterPointsSize = m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum * sizeof(CudaPoint2f);
    CUDA_CHECK(cudaMalloc((void**)&h_microImageParameterDevice.m_ppLensCenterPoints, lensCenterPointsSize));

    // 确保 m_microImageParameter.m_ppLensCenterPoints 是连续的
    CudaPoint2f* lensCenterPointsHost = new CudaPoint2f[m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum];
    for (int y = 0; y < m_rawImageParameter.m_yLensNum; ++y) {
        for (int x = 0; x < m_rawImageParameter.m_xLensNum; ++x) {
			//printf("%f,%f",m_microImageParameter.m_ppLensCenterPoints[y][x].x,m_microImageParameter.m_ppLensCenterPoints[y][x].y);
			
            lensCenterPointsHost[y * m_rawImageParameter.m_xLensNum + x].x = int(m_microImageParameter.m_ppLensCenterPoints[y][x].x);
			lensCenterPointsHost[y * m_rawImageParameter.m_xLensNum + x].y = int(m_microImageParameter.m_ppLensCenterPoints[y][x].y);
			//printf("x:%d,y:%d\n",lensCenterPointsHost[y * m_rawImageParameter.m_xLensNum + x].x,lensCenterPointsHost[y * m_rawImageParameter.m_xLensNum + x].y );
		}
    }
    CUDA_CHECK(cudaMemcpy(h_microImageParameterDevice.m_ppLensCenterPoints, lensCenterPointsHost, lensCenterPointsSize, cudaMemcpyHostToDevice));
    delete[] lensCenterPointsHost;

    // 为 m_ppPixelsMappingSet 分配内存并复制数据
    int pixelsMappingSetSize = m_rawImageParameter.m_srcImgHeight * m_rawImageParameter.m_srcImgWidth * sizeof(int);
    CUDA_CHECK(cudaMalloc((void**)&h_microImageParameterDevice.m_ppPixelsMappingSet, pixelsMappingSetSize));

    // 确保 m_microImageParameter.m_ppPixelsMappingSet 是连续的
    int* pixelsMappingSetHost = new int[m_rawImageParameter.m_srcImgHeight * m_rawImageParameter.m_srcImgWidth];
    for (int y = 0; y < m_rawImageParameter.m_srcImgHeight; ++y) {
        for (int x = 0; x < m_rawImageParameter.m_srcImgWidth; ++x) {
            pixelsMappingSetHost[y * m_rawImageParameter.m_srcImgWidth + x] = m_microImageParameter.m_ppPixelsMappingSet[y][x];
        }
    }
    CUDA_CHECK(cudaMemcpy(h_microImageParameterDevice.m_ppPixelsMappingSet, pixelsMappingSetHost, pixelsMappingSetSize, cudaMemcpyHostToDevice));
    delete[] pixelsMappingSetHost;

    // 为 m_ppMatchNeighborLens 分配内存并复制数据
    int matchNeighborLensSize = m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum * NEIGHBOR_MATCH_LENS_NUM * sizeof(MatchNeighborLens);
    CUDA_CHECK(cudaMalloc((void**)&h_microImageParameterDevice.m_ppMatchNeighborLens, matchNeighborLensSize));

    // 确保 m_microImageParameter.m_ppMatchNeighborLens 是连续的
    MatchNeighborLens* matchNeighborLensHost = new MatchNeighborLens[m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum * NEIGHBOR_MATCH_LENS_NUM];
    for (int y = 0; y < m_rawImageParameter.m_yLensNum; ++y) {
        for (int x = 0; x < m_rawImageParameter.m_xLensNum; ++x) {
            for (int n = 0; n < NEIGHBOR_MATCH_LENS_NUM; ++n) {
                matchNeighborLensHost[(y * m_rawImageParameter.m_xLensNum + x) * NEIGHBOR_MATCH_LENS_NUM + n] = m_microImageParameter.m_ppMatchNeighborLens[y][x][n];
            }
        }
    }
    CUDA_CHECK(cudaMemcpy(h_microImageParameterDevice.m_ppMatchNeighborLens, matchNeighborLensHost, matchNeighborLensSize, cudaMemcpyHostToDevice));
    delete[] matchNeighborLensHost;

	CUDA_CHECK(cudaMalloc((void**)&d_microImageParameter, sizeof(MicroImageParameterDevice)));
	
	// 将主机端的 h_microImageParameterDevice 拷贝到设备端的 d_microImageParameter
	CUDA_CHECK(cudaMemcpy(d_microImageParameter, &h_microImageParameterDevice, sizeof(MicroImageParameterDevice), cudaMemcpyHostToDevice));



	
	FilterParameterDevice h_filterPatameterDevice;

    // 2. 为 d_validNeighborPixelsNum 分配内存并复制数据
    int validNeighborPixelsNumSize = m_filterPatameter.m_pValidNeighborPixelsNum->total() * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&h_filterPatameterDevice.d_validNeighborPixelsNum, validNeighborPixelsNumSize));
    CUDA_CHECK(cudaMemcpy(h_filterPatameterDevice.d_validNeighborPixelsNum,
                          m_filterPatameter.m_pValidNeighborPixelsNum->ptr<float>(0),
                          validNeighborPixelsNumSize,
                          cudaMemcpyHostToDevice));

    // 3. 为 d_validPixelsMask 分配内存并复制数据
    int validPixelsMaskSize = m_filterPatameter.m_pValidPixelsMask->total() * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&h_filterPatameterDevice.d_validPixelsMask, validPixelsMaskSize));
    CUDA_CHECK(cudaMemcpy(h_filterPatameterDevice.d_validPixelsMask,
                          m_filterPatameter.m_pValidPixelsMask->ptr<float>(0),
                          validPixelsMaskSize,
                          cudaMemcpyHostToDevice));

    // 4. 为 d_filterKernel 分配内存并复制数据
    int filterKernelSize = m_filterPatameter.m_filterKnernel.total() * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&h_filterPatameterDevice.d_filterKernel, filterKernelSize));
    CUDA_CHECK(cudaMemcpy(h_filterPatameterDevice.d_filterKernel,
                          m_filterPatameter.m_filterKnernel.ptr<float>(0),
                          filterKernelSize,
                          cudaMemcpyHostToDevice));

    // 5. 为 d_filterPatameterDevice 分配内存
    CUDA_CHECK(cudaMalloc((void**)&d_filterPatameterDevice, sizeof(FilterParameterDevice)));

    // 6. 将主机端的 h_filterPatameterDevice 拷贝到设备端的 d_filterPatameterDevice
    CUDA_CHECK(cudaMemcpy(d_filterPatameterDevice, &h_filterPatameterDevice, sizeof(FilterParameterDevice), cudaMemcpyHostToDevice));



int numPatches = m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum;
RanderMapPatch* h_ppRanderMapPatch = new RanderMapPatch[numPatches];

// 2. 为每个 RanderMapPatch 的 simg 分配设备内存
float** h_simgDevice = new float*[numPatches];
for (int i = 0; i < numPatches; ++i) {
	CUDA_CHECK(cudaMalloc((void**)&h_simgDevice[i], DEST_WIDTH * DEST_HEIGHT * 3 * sizeof(float)));
	h_ppRanderMapPatch[i].sy = 0; // 示例值
	h_ppRanderMapPatch[i].sx = 0; // 示例值
	h_ppRanderMapPatch[i].simg = h_simgDevice[i];
}

// 3. 设备端分配 RanderMapPatch 数组
CUDA_CHECK(cudaMalloc((void**)&d_ppRanderMapPatch, numPatches * sizeof(RanderMapPatch)));

// 4. 将主机端的 RanderMapPatch 数组拷贝到设备端
CUDA_CHECK(cudaMemcpy(d_ppRanderMapPatch, h_ppRanderMapPatch, numPatches * sizeof(RanderMapPatch), cudaMemcpyHostToDevice));


dim3 block(16);
dim3 grid((m_rawImageParameter.m_xLensNum * m_rawImageParameter.m_yLensNum + block.x - 1) / block.x);
// Launch the kernel to print lens center points
printLensCenterPoints_1<<<grid, block>>>(d_microImageParameter, m_rawImageParameter.m_yLensNum, m_rawImageParameter.m_xLensNum);
// Check for any errors during kernel launch
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
}



void DataParameter::UpdateImgToGPU() {
    // 将图像从 CV_8UC3 转换为 CV_32FC3
    cv::Mat inputImgFloat, inputImgRecFloat;
    m_inputImg.convertTo(inputImgFloat, CV_32FC3, 1.0f / 255.0f);
    m_inputImgRec.convertTo(inputImgRecFloat, CV_32FC3, 1.0f / 255.0f);

    // 计算图像数据大小
    size_t inputImgSize = inputImgFloat.total() * inputImgFloat.elemSize();
    size_t inputImgRecSize = inputImgRecFloat.total() * inputImgRecFloat.elemSize();

    // 分配 GPU 内存
    CUDA_CHECK(cudaMalloc((void**)&d_inputImg, inputImgSize));
    CUDA_CHECK(cudaMalloc((void**)&d_inputImgRec, inputImgRecSize));

    // 将图像数据从主机复制到设备
    CUDA_CHECK(cudaMemcpy(d_inputImg, inputImgFloat.ptr<float>(0), inputImgSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inputImgRec, inputImgRecFloat.ptr<float>(0), inputImgRecSize, cudaMemcpyHostToDevice));
}

void saveSingleChannelGpuMemoryAsImage(float* d_data, int width, int height, const std::string& filename) {
    // 分配主机内存
    float* h_data = new float[width * height];

    // 从设备内存拷贝到主机内存
    CUDA_CHECK(cudaMemcpy(h_data, d_data, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    // 将浮点数据转换为 uchar 数据
    cv::Mat img(height, width, CV_32FC1, h_data);
    cv::Mat img_8u;
    img.convertTo(img_8u, CV_8UC1, 255.0); // 将浮点值 [0, 1] 转换为 [0, 255]

    // 保存图像
    cv::imwrite(filename, img_8u);

    // 释放主机内存
    delete[] h_data;
}

void saveThreeChannelGpuMemoryAsImage(float* d_data, int width, int height, const std::string& filename) {
    // 分配主机内存
    float* h_data = new float[width * height * 3];

    // 从设备内存拷贝到主机内存
    CUDA_CHECK(cudaMemcpy(h_data, d_data, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    // 将浮点数据转换为 uchar 数据
    cv::Mat img(height, width, CV_32FC3, h_data);
    cv::Mat img_8u;
    img.convertTo(img_8u, CV_8UC3, 255.0); // 将浮点值 [0, 1] 转换为 [0, 255]

    // 保存图像
    cv::imwrite(filename, img_8u);

    // 释放主机内存
    delete[] h_data;
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
		//��͹��ż���У���������������
		x_shift = -1;
	}
	else{
		//��͹�������У���������ż����
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