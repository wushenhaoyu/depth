#include "DataParameter.cuh"
using namespace cv;

using namespace std;


/*没有经验,蠢猪写法*/
__constant__ RawImageParameter d_rawImageParameter;
__constant__ DisparityParameter d_disparityParameter;
FilterParameterDevice* d_filterPatameterDevice; 
MicroImageParameterDevice* d_microImageParameter;
__device__ float* d_costVol;
__device__ float* d_costVolFiltered;
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
float* d_data; 




void DataParameter::mapToGPU() {
    /**************复制常量变量**************/
    CUDA_CHECK(cudaMemcpyToSymbol(d_rawImageParameter, &m_rawImageParameter, sizeof(RawImageParameter)));
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

    /**************分配内存并将指针传递给设备端**************/
    float* h_costVol;
    CUDA_CHECK(cudaMalloc((void**)&h_costVol, m_disparityParameter.m_disNum * m_rawImageParameter.m_recImgHeight * m_rawImageParameter.m_recImgWidth * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_costVol, &h_costVol, sizeof(float*)));

    float* h_costVolfilted;
    CUDA_CHECK(cudaMalloc((void**)&h_costVol, m_disparityParameter.m_disNum * m_rawImageParameter.m_recImgHeight * m_rawImageParameter.m_recImgWidth * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_costVolFiltered, &h_costVol, sizeof(float*)));

    CUDA_CHECK(cudaMalloc((void**)&d_rawDisp, m_rawImageParameter.m_recImgHeight * m_rawImageParameter.m_recImgWidth * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_rawDisp, 0, m_rawImageParameter.m_recImgHeight * m_rawImageParameter.m_recImgWidth * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&d_rawDisp_temp, m_rawImageParameter.m_recImgHeight * m_rawImageParameter.m_recImgWidth * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_rawDisp_temp, 0, m_rawImageParameter.m_recImgHeight * m_rawImageParameter.m_recImgWidth * sizeof(float)));

    float* h_ppLensMeanDisp;
    CUDA_CHECK(cudaMalloc((void**)&h_ppLensMeanDisp, m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_ppLensMeanDisp, &h_ppLensMeanDisp, sizeof(float*)));

    CUDA_CHECK(cudaMalloc((void**)&d_grayImg, m_rawImageParameter.m_srcImgHeight * m_rawImageParameter.m_srcImgWidth * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grayImg, 0, m_rawImageParameter.m_srcImgHeight * m_rawImageParameter.m_srcImgWidth * sizeof(float)));

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

    /**************初始化设备内存**************/
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

    /**************创建 MicroImageParameterDevice 实例**************/
    MicroImageParameterDevice h_microImageParameterDevice;
    h_microImageParameterDevice.m_circleDiameter = m_microImageParameter.m_circleDiameter;
    h_microImageParameterDevice.m_circleNarrow = m_microImageParameter.m_circleNarrow;
    h_microImageParameterDevice.m_radiusDisEqu = m_microImageParameter.m_radiusDisEqu;

    /**************为 m_ppLensCenterPoints 分配内存并复制数据**************/
    int lensCenterPointsSize = m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum * sizeof(CudaPoint2f);
    CUDA_CHECK(cudaMalloc((void**)&h_microImageParameterDevice.m_ppLensCenterPoints, lensCenterPointsSize));

    CudaPoint2f* lensCenterPointsHost = new CudaPoint2f[m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum];
    for (int y = 0; y < m_rawImageParameter.m_yLensNum; ++y) {
        for (int x = 0; x < m_rawImageParameter.m_xLensNum; ++x) {
            lensCenterPointsHost[y * m_rawImageParameter.m_xLensNum + x].x = m_microImageParameter.m_ppLensCenterPoints[y][x].x;
            lensCenterPointsHost[y * m_rawImageParameter.m_xLensNum + x].y = m_microImageParameter.m_ppLensCenterPoints[y][x].y;
        }
    }
    CUDA_CHECK(cudaMemcpy(h_microImageParameterDevice.m_ppLensCenterPoints, lensCenterPointsHost, lensCenterPointsSize, cudaMemcpyHostToDevice));
    delete[] lensCenterPointsHost;

    /**************为 m_ppPixelsMappingSet 分配内存并复制数据**************/
    int pixelsMappingSetSize = m_rawImageParameter.m_srcImgHeight * m_rawImageParameter.m_srcImgWidth * sizeof(int);
    CUDA_CHECK(cudaMalloc((void**)&h_microImageParameterDevice.m_ppPixelsMappingSet, pixelsMappingSetSize));

    int* pixelsMappingSetHost = new int[m_rawImageParameter.m_srcImgHeight * m_rawImageParameter.m_srcImgWidth];
    for (int y = 0; y < m_rawImageParameter.m_srcImgHeight; ++y) {
        for (int x = 0; x < m_rawImageParameter.m_srcImgWidth; ++x) {
            pixelsMappingSetHost[y * m_rawImageParameter.m_srcImgWidth + x] = m_microImageParameter.m_ppPixelsMappingSet[y][x];
        }
    }
    CUDA_CHECK(cudaMemcpy(h_microImageParameterDevice.m_ppPixelsMappingSet, pixelsMappingSetHost, pixelsMappingSetSize, cudaMemcpyHostToDevice));
    delete[] pixelsMappingSetHost;

    /**************为 m_ppMatchNeighborLens 分配内存并复制数据**************/
    int matchNeighborLensSize = m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum * NEIGHBOR_MATCH_LENS_NUM * sizeof(MatchNeighborLens);
    CUDA_CHECK(cudaMalloc((void**)&h_microImageParameterDevice.m_ppMatchNeighborLens, matchNeighborLensSize));

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
    CUDA_CHECK(cudaMemcpy(d_microImageParameter, &h_microImageParameterDevice, sizeof(MicroImageParameterDevice), cudaMemcpyHostToDevice));

    /**************FilterParameterDevice 数据复制**************/
    FilterParameterDevice h_filterPatameterDevice;

    int validNeighborPixelsNumSize = m_filterPatameter.m_pValidNeighborPixelsNum->total() * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&h_filterPatameterDevice.d_validNeighborPixelsNum, validNeighborPixelsNumSize));
    CUDA_CHECK(cudaMemcpy(h_filterPatameterDevice.d_validNeighborPixelsNum,
                          m_filterPatameter.m_pValidNeighborPixelsNum->ptr<float>(0),
                          validNeighborPixelsNumSize,
                          cudaMemcpyHostToDevice));

    int validPixelsMaskSize = m_filterPatameter.m_pValidPixelsMask->total() * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&h_filterPatameterDevice.d_validPixelsMask, validPixelsMaskSize));
    CUDA_CHECK(cudaMemcpy(h_filterPatameterDevice.d_validPixelsMask,
                          m_filterPatameter.m_pValidPixelsMask->ptr<float>(0),
                          validPixelsMaskSize,
                          cudaMemcpyHostToDevice));

    int filterKernelSize = m_filterPatameter.m_filterKnernel.total() * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&h_filterPatameterDevice.d_filterKernel, filterKernelSize));
    CUDA_CHECK(cudaMemcpy(h_filterPatameterDevice.d_filterKernel,
                          m_filterPatameter.m_filterKnernel.ptr<float>(0),
                          filterKernelSize,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_filterPatameterDevice, sizeof(FilterParameterDevice)));
    CUDA_CHECK(cudaMemcpy(d_filterPatameterDevice, &h_filterPatameterDevice, sizeof(FilterParameterDevice), cudaMemcpyHostToDevice));

    /**************为 RanderMapPatch 分配内存**************/
    int numPatches = m_rawImageParameter.m_yLensNum * m_rawImageParameter.m_xLensNum;
    RanderMapPatch* h_ppRanderMapPatch = new RanderMapPatch[numPatches];

    float** h_simgDevice = new float*[numPatches];
    for (int i = 0; i < numPatches; ++i) {
        CUDA_CHECK(cudaMalloc((void**)&h_simgDevice[i], DEST_WIDTH * DEST_HEIGHT * 3 * sizeof(float)));
        h_ppRanderMapPatch[i].sy = 0; 
        h_ppRanderMapPatch[i].sx = 0;
        h_ppRanderMapPatch[i].simg = h_simgDevice[i];
    }

    CUDA_CHECK(cudaMalloc((void**)&d_ppRanderMapPatch, numPatches * sizeof(RanderMapPatch)));
    CUDA_CHECK(cudaMemcpy(d_ppRanderMapPatch, h_ppRanderMapPatch, numPatches * sizeof(RanderMapPatch), cudaMemcpyHostToDevice));

}
PvString lConnectionID;
PvResult lResult;
PvDevice* lDevice = nullptr;
PvStream* lStream = nullptr;
PvDeviceGEV* lDeviceGEV = nullptr;
PvPipeline* lPipeline = nullptr;
PvGenParameterArray* lDeviceParams = nullptr;
PvGenCommand* lStart = nullptr;
PvGenCommand* lStop = nullptr;
PvBuffer* lBuffer = nullptr;
PvResult lOperationResult;
// void DataParameter::UploadtoGPU() const
// {
// 	lBuffer = NULL;
//     lOperationResult;
// 	lResult = lPipeline->RetrieveNextBuffer(&lBuffer, 1000, &lOperationResult);
//         if (lResult.IsOK() && lOperationResult.IsOK()) {
// 			PvImage *lImage = lBuffer->GetImage();
// 			uint32_t lWidth = lImage->GetWidth();
// 			uint32_t lHeight = lImage->GetHeight();
// 			PvPixelType lPixelType = lImage->GetPixelType();

// 			Mat frame;
// 			if (lPixelType == PvPixelMono8) {
// 				frame = Mat(lHeight, lWidth, CV_8UC3, lImage->GetDataPointer());
// 			} else {
// 				cout << "Unsupported pixel type: " << lPixelType << endl;
// 				lPipeline->ReleaseBuffer(lBuffer);
// 			}
// 			rotate(frame, frame, ROTATE_90_COUNTERCLOCKWISE);

// 			Mat inputImgRec = frame(Rect(
// 				m_rawImageParameter.m_xPixelBeginOffset,
// 				m_rawImageParameter.m_yPixelBeginOffset,
// 				m_rawImageParameter.m_recImgWidth,
// 				m_rawImageParameter.m_recImgHeight
// 			)).clone();
// 			imshow("Original", frame);

// 			Mat inputImgRecFloat,inputImgFloat;
// 			inputImgRec.convertTo(inputImgRecFloat, CV_32FC3, 1.0f / 255.0f);
// 			frame.convertTo(inputImgFloat, CV_32FC3, 1.0f / 255.0f);
// 			size_t inputImgRecSize = inputImgRecFloat.total() * inputImgRecFloat.elemSize();
// 			CUDA_CHECK(cudaMalloc((void**)&d_inputImgRec, inputImgRecSize));
// 			CUDA_CHECK(cudaMemcpy(d_inputImgRec, inputImgRecFloat.ptr<float>(0), inputImgRecSize, cudaMemcpyHostToDevice));
// 			size_t inputImgSize = inputImgFloat.total() * inputImgFloat.elemSize();
// 			CUDA_CHECK(cudaMalloc((void**)&d_inputImg, inputImgSize));
// 			CUDA_CHECK(cudaMemcpy(d_inputImg, inputImgFloat.ptr<float>(0), inputImgSize, cudaMemcpyHostToDevice));
// 		}
// }


void DataParameter::UploadtoGPU() const
{
    lBuffer = NULL;
    lOperationResult;
    lResult = lPipeline->RetrieveNextBuffer(&lBuffer, 1000, &lOperationResult);

    if (lResult.IsOK() && lOperationResult.IsOK()) {
        PvImage *lImage = lBuffer->GetImage();
        uint32_t lWidth = lImage->GetWidth();
        uint32_t lHeight = lImage->GetHeight();
        PvPixelType lPixelType = lImage->GetPixelType();

        Mat frame;
        if (lPixelType == PvPixelMono8) {
            // 1. 正确地以灰度图格式读取
            Mat gray(lHeight, lWidth, CV_8UC1, lImage->GetDataPointer());

            // 2. 转换为三通道灰度图
            cvtColor(gray, frame, COLOR_GRAY2BGR);
        } else {
            cout << "Unsupported pixel type: " << lPixelType << endl;
            lPipeline->ReleaseBuffer(lBuffer);
            return;  // 加上 return 避免之后继续执行
        }

        // 图像旋转
        rotate(frame, frame, ROTATE_90_COUNTERCLOCKWISE);

        // 裁剪区域
        Mat inputImgRec = frame(Rect(
            m_rawImageParameter.m_xPixelBeginOffset,
            m_rawImageParameter.m_yPixelBeginOffset,
            m_rawImageParameter.m_recImgWidth,
            m_rawImageParameter.m_recImgHeight
        )).clone();

        // 显示图像
        imshow("Original", frame);


        // 转换为 float 并上传 GPU
        Mat inputImgRecFloat, inputImgFloat;
        inputImgRec.convertTo(inputImgRecFloat, CV_32FC3, 1.0f / 255.0f);
        frame.convertTo(inputImgFloat, CV_32FC3, 1.0f / 255.0f);

        size_t inputImgRecSize = inputImgRecFloat.total() * inputImgRecFloat.elemSize();
        CUDA_CHECK(cudaMalloc((void**)&d_inputImgRec, inputImgRecSize));
        CUDA_CHECK(cudaMemcpy(d_inputImgRec, inputImgRecFloat.ptr<float>(0), inputImgRecSize, cudaMemcpyHostToDevice));

        size_t inputImgSize = inputImgFloat.total() * inputImgFloat.elemSize();
        CUDA_CHECK(cudaMalloc((void**)&d_inputImg, inputImgSize));
        CUDA_CHECK(cudaMemcpy(d_inputImg, inputImgFloat.ptr<float>(0), inputImgSize, cudaMemcpyHostToDevice));
		lPipeline->ReleaseBuffer(lBuffer);
    }
	
}


void DataParameter::UpdateImgToGPU() {
	if (!PvSelectDevice(&lConnectionID)) {
        cout << "No device selected" << endl;
        //PV_SAMPLE_TERMINATE();
        return;
    }
	lDevice = PvDevice::CreateAndConnect(lConnectionID, &lResult);
	lDeviceParams = lDevice->GetParameters();
		// 禁用 TestPattern 功能
		PvGenEnum* lTestPattern = dynamic_cast<PvGenEnum*>(lDeviceParams->Get("TestPattern"));
		if (lTestPattern != nullptr)
		{
			lTestPattern->SetValue("Off");
		}
		
		// 设置图像宽度和高度
		lResult = lDeviceParams->SetIntegerValue("Width", 1920);
		if (!lResult.IsOK())
		{
			cout << "Failed to set Width: " << lResult.GetCodeString().GetAscii() << endl;
		}
		
		lResult = lDeviceParams->SetIntegerValue("Height", 1080);
		if (!lResult.IsOK())
		{
			cout << "Failed to set Height: " << lResult.GetCodeString().GetAscii() << endl;
		}
    if (!lDevice || !lResult.IsOK()) {
        cout << "Unable to connect to device: " << lResult.GetCodeString().GetAscii() << endl;
        PvDevice::Free(lDevice);
        //PV_SAMPLE_TERMINATE();
        return;
    }
	lStream = PvStream::CreateAndOpen(lConnectionID, &lResult);
    if (!lStream || !lResult.IsOK()) {
        cout << "Unable to open stream: " << lResult.GetCodeString().GetAscii() << endl;
        PvDevice::Free(lDevice);
        //PV_SAMPLE_TERMINATE();
        return;
    }
	lDeviceGEV = dynamic_cast<PvDeviceGEV *>(lDevice);
    if (lDeviceGEV) {
        PvStreamGEV *lStreamGEV = dynamic_cast<PvStreamGEV *>(lStream);
        if (lStreamGEV) {
            lDeviceGEV->NegotiatePacketSize();
            lDeviceGEV->SetStreamDestination(lStreamGEV->GetLocalIPAddress(), lStreamGEV->GetLocalPort());
        }
    }
	lPipeline = new PvPipeline(lStream);
    if (lPipeline) {
        uint32_t lSize = lDevice->GetPayloadSize();
        lPipeline->SetBufferCount(BUFFER_COUNT);
        lPipeline->SetBufferSize(lSize);
    } else {
        cout << "Failed to create pipeline" << endl;
        lStream->Close();
        PvStream::Free(lStream);
        PvDevice::Free(lDevice);
        //PV_SAMPLE_TERMINATE();
        return;
    }
	PvGenEnum *lPixelFormat = dynamic_cast<PvGenEnum *>(lDeviceParams->Get("PixelFormat"));
	int64_t lWidth = 1920, lHeight = 1080;
	PvGenCommand* lAcquisitionStop = dynamic_cast<PvGenCommand*>(lDeviceParams->Get("AcquisitionStop"));
	if (lAcquisitionStop != nullptr)
	{
		lAcquisitionStop->Execute();
	}
	


	lDeviceParams->GetIntegerValue( "Width", lWidth );
    lDeviceParams->GetIntegerValue( "Height", lHeight );
	lStart = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStart"));
    lStop = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStop"));
	lPipeline->Start();
    lDevice->StreamEnable();
    lStart->Execute();

	lBuffer = NULL;
    lOperationResult;

	lResult = lPipeline->RetrieveNextBuffer(&lBuffer, 1000, &lOperationResult);
        if (lResult.IsOK() && lOperationResult.IsOK()) {
			PvImage *lImage = lBuffer->GetImage();
			uint32_t lWidth = lImage->GetWidth();
			uint32_t lHeight = lImage->GetHeight();

			PvPixelType lPixelType = lImage->GetPixelType();

			Mat frame;
			if (lPixelType == PvPixelMono8) {
				frame = Mat(lHeight, lWidth, CV_8UC1, lImage->GetDataPointer());
				cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
			} else {
				cout << "Unsupported pixel type: " << lPixelType << endl;
				lPipeline->ReleaseBuffer(lBuffer);
			}
			imwrite("tt.bmp", frame);
			rotate(frame, frame, ROTATE_90_COUNTERCLOCKWISE);
			frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
			//保存图像
			Mat inputImgRec = frame(Rect(
				m_rawImageParameter.m_xPixelBeginOffset,
				m_rawImageParameter.m_yPixelBeginOffset,
				m_rawImageParameter.m_recImgWidth,
				m_rawImageParameter.m_recImgHeight
			)).clone();

			Mat inputImgRecFloat,inputImgFloat;
			inputImgRec.convertTo(inputImgRecFloat, CV_32FC3, 1.0f / 255.0f);
			frame.convertTo(inputImgFloat, CV_32FC3, 1.0f / 255.0f);
			size_t inputImgRecSize = inputImgRecFloat.total() * inputImgRecFloat.elemSize();
			CUDA_CHECK(cudaMalloc((void**)&d_inputImgRec, inputImgRecSize));
			CUDA_CHECK(cudaMemcpy(d_inputImgRec, inputImgRecFloat.ptr<float>(0), inputImgRecSize, cudaMemcpyHostToDevice));
			size_t inputImgSize = inputImgFloat.total() * inputImgFloat.elemSize();
			CUDA_CHECK(cudaMalloc((void**)&d_inputImg, inputImgSize));
			CUDA_CHECK(cudaMemcpy(d_inputImg, inputImgFloat.ptr<float>(0), inputImgSize, cudaMemcpyHostToDevice));
			cv::Mat im_gray,im_grad;
			cv::cvtColor(inputImgRec, im_gray, cv::COLOR_RGB2GRAY);
			cv::Sobel(im_gray, im_grad, CV_32F, 1, 0, 1);
			im_grad += 0.5;
			size_t im_grad_size = im_grad.total() * im_grad.elemSize();
			CUDA_CHECK(cudaMalloc((void**)&d_gradImg, im_grad_size));
			CUDA_CHECK(cudaMemcpy(d_gradImg, im_grad.ptr<float>(0), im_grad_size, cudaMemcpyHostToDevice));

		}

	
	// 将图像从 CV_8UC3 转换为 CV_32FC3
    /*cv::Mat inputImgFloat, inputImgRecFloat;
	cv::Mat m_inputImg_,m_gradImg;
	m_inputImg.convertTo(m_inputImg_, CV_32FC3, 1 / 255.0f);
    cv::Mat im_gray, tmp;
    m_inputImg_.convertTo(tmp, CV_32F);
    cv::cvtColor(tmp, im_gray, COLOR_RGB2GRAY);

    cv::Mat dst_x, dst_y;
    cv::Sobel(im_gray, m_gradImg, CV_32F, 1, 0, 1);
    m_gradImg += 0.5;
    m_inputImg.convertTo(inputImgFloat, CV_32FC3, 1.0f / 255.0f);
    m_inputImgRec.convertTo(inputImgRecFloat, CV_32FC3, 1.0f / 255.0f);

	size_t im_grad_size = m_gradImg.total() * m_gradImg.elemSize();
	CUDA_CHECK(cudaMalloc((void**)&d_gradImg, im_grad_size));
	CUDA_CHECK(cudaMemcpy(d_gradImg ,m_gradImg.ptr<float>(0), im_grad_size, cudaMemcpyHostToDevice));


    // 计算图像数据大小
    size_t inputImgSize = inputImgFloat.total() * inputImgFloat.elemSize();
    size_t inputImgRecSize = inputImgRecFloat.total() * inputImgRecFloat.elemSize();


    // 分配 GPU 内存
    CUDA_CHECK(cudaMalloc((void**)&d_inputImg, inputImgSize));
    CUDA_CHECK(cudaMalloc((void**)&d_inputImgRec, inputImgRecSize));

    // 将图像数据从主机复制到设备
    CUDA_CHECK(cudaMemcpy(d_inputImg, inputImgFloat.ptr<float>(0), inputImgSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inputImgRec, inputImgRecFloat.ptr<float>(0), inputImgRecSize, cudaMemcpyHostToDevice));*/
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