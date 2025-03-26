#include "ImageRander.h"
#include "DataParameter.cuh"
#include <iomanip>

using namespace std;
using namespace cv;


//修改了WIDTH和HEIGHT,此处参数只能影响到渲染图的生成效果
#define MEAN_DISP_LEN_RADIUS 18//平均距离长度 8 注意该参数在需要让计算深度的点尽量都在一个圆内(方型)  10
#define PATCH_SCALE9 9//路径比例 9
#define RANDER_SCALE 0.9//渲染比例 render  0.35
#define DEST_WIDTH 44//38 27 44
#define DEST_HEIGHT 44//38 27 44

ImageRander::ImageRander()
{

}

ImageRander::~ImageRander()
{

}

void ImageRander::imageRanderWithMask(const DataParameter &dataParameter, cv::Mat &rawDisp, cv::Mat *confidentMask)
{//对带置信度mask的情况进行子孔径渲染
	RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
	MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
	DisparityParameter disparityParameter = dataParameter.getDisparityParameter();

	cv::Mat tmpRawDisp = cv::Mat::zeros(rawDisp.rows, rawDisp.cols, CV_32FC1);
	rawDisp.convertTo(tmpRawDisp, CV_32F, disparityParameter.m_dispStep, disparityParameter.m_dispMin);

	float **ppLensMeanDisp = new float*[rawImageParameter.m_yLensNum];
	for (int i = 0; i < rawImageParameter.m_yLensNum; i++)
		ppLensMeanDisp[i] = new float[rawImageParameter.m_xLensNum];

//#pragma omp parallel for
	for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
	{
		for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
		{
			Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];
			int x_begin = curCenterPos.x - rawImageParameter.m_xPixelBeginOffset - MEAN_DISP_LEN_RADIUS;
			int y_begin = curCenterPos.y - rawImageParameter.m_yPixelBeginOffset - MEAN_DISP_LEN_RADIUS;

			cv::Mat srcCost = tmpRawDisp(cv::Rect(x_begin, y_begin, MEAN_DISP_LEN_RADIUS * 2 + 1, MEAN_DISP_LEN_RADIUS * 2 + 1));
			cv::Mat mask = (*confidentMask)(cv::Rect(x_begin, y_begin, MEAN_DISP_LEN_RADIUS * 2 + 1, MEAN_DISP_LEN_RADIUS * 2 + 1));
			ppLensMeanDisp[y][x] = std::max(cv::mean(srcCost, mask)[0], (double)(disparityParameter.m_dispMin));
		}
	}

	//渲染开始
	cv::Mat randerMapinput, randerSceneMap, finalRanderMap;
	//dataParameter.m_inputImgRec.convertTo(randerMapinput, CV_64FC3);
	dataParameter.m_inputImgRec.convertTo(randerMapinput, CV_64FC1);//转为双精度浮点数单通道
	imageRander(ppLensMeanDisp, rawImageParameter, microImageParameter, randerMapinput, randerSceneMap);
	std::string storeName = dataParameter.m_folderPath + "/randerSceneMap.bmp";
	//randerSceneMap.convertTo(finalRanderMap, CV_8UC3);
	randerSceneMap.convertTo(finalRanderMap, CV_8UC1);
	imwrite(storeName, finalRanderMap);

	//这部分用来渲染深度图像，测试速度阶段暂时注释掉
//	/*
	cv::Mat sceneDisp = cv::Mat::zeros(rawDisp.rows, rawDisp.cols, CV_64FC1);
	rawDisp.convertTo(sceneDisp, CV_64FC1, disparityParameter.m_dispStep, disparityParameter.m_dispMin);
	cv::Mat randerDispMap, randerSparseDispMap;
	imageRander(ppLensMeanDisp, rawImageParameter, microImageParameter, sceneDisp, randerDispMap);
	randerDispMap.copyTo(randerSparseDispMap);
	storeName = dataParameter.m_folderPath + "/randerDispMap.bmp";
	dispMapShowForColor(storeName, randerDispMap);

	cv::Mat rawMask, randerSceneMask;
	(*confidentMask).convertTo(rawMask, CV_64FC1);
	imageRander(ppLensMeanDisp, rawImageParameter, microImageParameter, rawMask, randerSceneMask);
	cv::Mat finalRanderMask = cv::Mat::zeros(randerSceneMask.rows, randerSceneMask.cols, CV_8UC1);
//#pragma omp parallel for
	for (int py = 0; py < randerSceneMask.rows; py++)
	{
		double *ySceneData = (double *)randerSceneMask.ptr<double>(py);
		uchar *yFinalData = (uchar *)finalRanderMask.ptr<uchar>(py);
		for (int px = 0; px < randerSceneMask.cols; px++)
		{
			if (ySceneData[px] >= 255.0)
				yFinalData[px] = 255;
		}
	}
	storeName = dataParameter.m_folderPath + "/randerSceneMask.bmp";
	imwrite(storeName, finalRanderMask);

	outputSparseSceneDepth(dataParameter.m_folderPath, randerSparseDispMap, finalRanderMask);

	double minVal, maxVal;
	minMaxLoc(randerSparseDispMap, &minVal, &maxVal);
//#pragma omp parallel for
	for (int py = 0; py < randerSparseDispMap.rows; py++)
	{
		double *ySparseData = (double *)randerSparseDispMap.ptr<double>(py);
		uchar *yConfMask = (uchar *)finalRanderMask.ptr<uchar>(py);
		for (int px = 0; px < randerSparseDispMap.cols; px++)
		{
			if (yConfMask[px] == 0)
				ySparseData[px] = minVal;
		}
	}
	storeName = dataParameter.m_folderPath + "/randerSparseDispMap.bmp";
	dispMapShowForColor(storeName, randerSparseDispMap);
//  */
	for (int i = 0; i < rawImageParameter.m_yLensNum; i++)
		delete[]ppLensMeanDisp[i];
	delete[]ppLensMeanDisp;
}

extern __constant__ RawImageParameter d_rawImageParameter;
extern __constant__ DisparityParameter d_disparityParameter;
extern __constant__ FilterParameterDevice d_filterPatameterDevice; 
extern __device__ MicroImageParameterDevice d_microImageParameter; 
extern __device__ float* d_costVol;
extern __device__ float* d_rawDisp;
extern __device__ float* d_ppLensMeanDisp;

__global__ void computeLensMeanDispKernel()
{
    // 获取当前线程的坐标
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保线程在有效范围内
    if (y >= d_rawImageParameter.m_yCenterBeginOffset && y < d_rawImageParameter.m_yLensNum - d_rawImageParameter.m_yCenterEndOffset &&
        x >= d_rawImageParameter.m_xCenterBeginOffset && x < d_rawImageParameter.m_xLensNum - d_rawImageParameter.m_xCenterEndOffset)
    {
        // 计算当前中心点坐标
        Point2d curCenterPos = d_microImageParameter.m_ppLensCenterPoints[y * d_rawImageParameter.m_xLensNum + x];
        int x_begin = curCenterPos.x - d_rawImageParameter.m_xPixelBeginOffset - MEAN_DISP_LEN_RADIUS;
        int y_begin = curCenterPos.y - d_rawImageParameter.m_yPixelBeginOffset - MEAN_DISP_LEN_RADIUS;
        
        // 设置共享内存
        extern __shared__ float sharedMem[];

        // 计算共享内存的宽度和高度
        int rectWidth = MEAN_DISP_LEN_RADIUS * 2 + 1;
        int rectHeight = MEAN_DISP_LEN_RADIUS * 2 + 1;

        // 将区域数据加载到共享内存
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int sharedIndex = ty * rectWidth + tx;

        // 加载共享内存的每个元素
        if (x_begin + tx >= 0 && x_begin + tx < d_rawImageParameter.m_recImgWidth &&
            y_begin + ty >= 0 && y_begin + ty < d_rawImageParameter.m_recImgHeight)
        {
            sharedMem[sharedIndex] = d_rawDisp[(y_begin + ty) * d_rawImageParameter.m_recImgWidth + (x_begin + tx)];
        }
        else
        {
            sharedMem[sharedIndex] = 0.0f;  // 边界外的数据填充为0
        }

        // 确保所有线程都完成数据加载
        __syncthreads();

        // 计算该区域的均值
        float sum = 0.0f;
        int count = 0;

        // 计算共享内存中的区域均值
        for (int dy = 0; dy < rectHeight; dy++) {
            for (int dx = 0; dx < rectWidth; dx++) {
                int sharedIndex = (dy * rectWidth) + dx;
                sum += sharedMem[sharedIndex];
                count++;
            }
        }

        float meanDisp = sum / count;
        d_ppLensMeanDisp[y * d_rawImageParameter.m_xLensNum + x] = fmax(meanDisp, (float)d_disparityParameter.m_dispMin);
    }
}


void ImageRander::imageRanderWithOutMask(const DataParameter &dataParameter, cv::Mat &rawDisp)
{
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();

    // Define block and grid sizes
    dim3 blockSize(32, 32);
    dim3 gridSize((rawImageParameter.m_xLensNum + blockSize.x - 1) / blockSize.x, 
                  (rawImageParameter.m_yLensNum + blockSize.y - 1) / blockSize.y);

    computeLensMeanDispKernel<<<gridSize, blockSize>>>();

    // Check for any errors during kernel launch
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in kernel launch: " << cudaGetErrorString(err) << std::endl;
    }

	cv::Mat randerMapinput, randerSceneMap, finalRanderMap;
	dataParameter.m_inputImgRec.convertTo(randerMapinput, CV_64FC3);
	imageRander(nullptr, rawImageParameter, microImageParameter, randerMapinput, randerSceneMap);
	std::string storeName = dataParameter.m_folderPath + "/randerSceneMap.bmp";
	randerSceneMap.convertTo(finalRanderMap, CV_8UC3);
	//imwrite(storeName, finalRanderMap);
}

__global__ void resizeKernel(float *d_src, float *d_dst, int src_width, int src_height, int dst_width, int dst_height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dst_width && y < dst_height)
    {
        // 计算源图像对应位置的坐标
        float fx = (float)x / (dst_width - 1) * (src_width - 1);
        float fy = (float)y / (dst_height - 1) * (src_height - 1);
        
        int ix = (int)fx;
        int iy = (int)fy;

        // 获取邻域的像素值并执行双线性插值
        float wx = fx - ix;
        float wy = fy - iy;

        for (int c = 0; c < channels; ++c)
        {
            float top_left = d_src[(iy * src_width + ix) * channels + c];
            float top_right = d_src[(iy * src_width + (ix + 1)) * channels + c];
            float bottom_left = d_src[((iy + 1) * src_width + ix) * channels + c];
            float bottom_right = d_src[((iy + 1) * src_width + (ix + 1)) * channels + c];

            float interpolated = (1 - wx) * (1 - wy) * top_left +
                                 wx * (1 - wy) * top_right +
                                 (1 - wx) * wy * bottom_left +
                                 wx * wy * bottom_right;

            d_dst[(y * dst_width + x) * channels + c] = interpolated;
        }
    }
}

void resizeCUDA(float *d_src, float *d_dst, int src_width, int src_height, int dst_width, int dst_height, int channels)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((dst_width + blockSize.x - 1) / blockSize.x, (dst_height + blockSize.y - 1) / blockSize.y);

    resizeKernel<<<gridSize, blockSize>>>(d_src, d_dst, src_width, src_height, dst_width, dst_height, channels);
    cudaDeviceSynchronize();
}

__global__ void flipKernel(float *d_src, float *d_dst, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // 水平翻转：交换左右位置
        int flip_x = width - 1 - x;

        for (int c = 0; c < channels; ++c)
        {
            d_dst[(y * width + x) * channels + c] = d_src[(y * width + flip_x) * channels + c];
        }
    }
}

void flipCUDA(float *d_src, float *d_dst, int width, int height, int channels)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    flipKernel<<<gridSize, blockSize>>>(d_src, d_dst, width, height, channels);
    cudaDeviceSynchronize();
}


__global__ void accumulateKernel(
    float *d_randerMap, 
    float *d_randerCount, 
    RanderMapPatch *d_randerMapPatch, // 使用单一一级指针
    int m_xLensNum, 
    int m_yLensNum, 
    int DEST_WIDTH_, 
    int DEST_HEIGHT_, 
    int randerMapWidth, 
    int randerMapHeight, 
    int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < m_xLensNum && y < m_yLensNum)
    {
        // 使用线性索引来访问 RanderMapPatch
        RanderMapPatch patch = d_randerMapPatch[y * m_xLensNum + x];

        // Calculate the starting points for the patch in the output image
        int sy_b = patch.sy - DEST_HEIGHT_ / 2;
        int sx_b = patch.sx - DEST_WIDTH_ / 2;

        // Process the patch
        for (int py = 0; py < DEST_HEIGHT_; ++py)
        {
            for (int px = 0; px < DEST_WIDTH_; ++px)
            {
                int rander_x = sx_b + px;
                int rander_y = sy_b + py;

                if (rander_x >= 0 && rander_x < randerMapWidth && rander_y >= 0 && rander_y < randerMapHeight)
                {
                    // Accumulate the patch into the render map
                    for (int c = 0; c < channels; ++c)
                    {
                        atomicAdd(&d_randerMap[(rander_y * randerMapWidth + rander_x) * channels + c], patch.simg[(py * DEST_WIDTH_ + px) * channels + c]);
                    }

                    // Count how many patches contributed to each pixel
                    atomicAdd(&d_randerCount[rander_y * randerMapWidth + rander_x], 1.0f);
                }
            }
        }
    }
}


__global__ void normalizeKernel(float *d_randerMap, float *d_randerCount, int randerMap_width, int randerMap_height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < randerMap_width && y < randerMap_height)
    {
        for (int c = 0; c < channels; ++c)
        {
            int idx = (y * randerMap_width + x) * channels + c;
            if (d_randerCount[y * randerMap_width + x] > 0)
            {
                d_randerMap[idx] /= d_randerCount[y * randerMap_width + x];
            }
        }
    }
}


__global__ void computeBoundaryKernel(RanderMapPatch *d_ppRanderMapPatch, 
    const RawImageParameter rawImageParameter, 
    int *sx_begin, int *sy_begin, 
    int *sx_end, int *sy_end, 
    int DEST_WIDTH_, int DEST_HEIGHT_)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < rawImageParameter.m_xLensNum && y < rawImageParameter.m_yLensNum)
    {
        // 使用一维数组访问
        int sy = d_ppRanderMapPatch[y * rawImageParameter.m_xLensNum + x].sy;
        int sx = d_ppRanderMapPatch[y * rawImageParameter.m_xLensNum + x].sx;

        atomicMin(sx_begin, sx - DEST_WIDTH_ / 2);
        atomicMin(sy_begin, sy - DEST_HEIGHT_ / 2);
        atomicMax(sx_end, sx + DEST_WIDTH_ / 2);
        atomicMax(sy_end, sy + DEST_HEIGHT_ / 2);
    }
}


__global__ void processPatchKernel(
    float *d_randerImg, float *d_tmp, float *d_simg,
    const RawImageParameter rawImageParameter,
    const MicroImageParameter microImageParameter,
    float **ppLensMeanDisp,
    int offsetX, int offsetY, int patchWidth, int patchHeight, int channels,
    RanderMapPatch *d_ppRanderMapPatch)  // Changed to pointer to a flat array
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < rawImageParameter.m_xLensNum && y < rawImageParameter.m_yLensNum)
    {
        // Step 1: Get lens center position (sx, sy) from MicroImageParameter
        Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];

        // Calculate the patch block size based on lens displacement (using ppLensMeanDisp)
        int blockSize = fabsf(roundf(d_ppLensMeanDisp[y * rawImageParameter.m_xLensNum + x]));

        // Step 2: Define the start coordinates of the patch (startx, starty)
        int starty = curCenterPos.y - blockSize / 2 - rawImageParameter.m_yPixelBeginOffset;
        int startx = curCenterPos.x - blockSize / 2 - rawImageParameter.m_xPixelBeginOffset;

        // Step 3: Resize the patch using bilinear interpolation
        float *d_srcImg = d_randerImg + (starty * rawImageParameter.m_xLensNum + startx) * channels;

        // Resize using bilinear interpolation
        for (int j = 0; j < patchHeight; ++j)
        {
            for (int i = 0; i < patchWidth; ++i)
            {
                float fx = (float)i / (patchWidth - 1) * (blockSize - 1);
                float fy = (float)j / (patchHeight - 1) * (blockSize - 1);

                int ix = (int)fx;
                int iy = (int)fy;

                // Perform bilinear interpolation
                float wx = fx - ix;
                float wy = fy - iy;

                for (int c = 0; c < channels; ++c)
                {
                    float top_left = d_srcImg[(iy * rawImageParameter.m_xLensNum + ix) * channels + c];
                    float top_right = d_srcImg[(iy * rawImageParameter.m_xLensNum + ix + 1) * channels + c];
                    float bottom_left = d_srcImg[((iy + 1) * rawImageParameter.m_xLensNum + ix) * channels + c];
                    float bottom_right = d_srcImg[((iy + 1) * rawImageParameter.m_xLensNum + ix + 1) * channels + c];

                    float interpolated = (1 - wx) * (1 - wy) * top_left +
                                         wx * (1 - wy) * top_right +
                                         (1 - wx) * wy * bottom_left +
                                         wx * wy * bottom_right;

                    d_tmp[(j * patchWidth + i) * channels + c] = interpolated;
                }
            }
        }

        // Step 4: Flip the patch horizontally (or vertically if needed)
        for (int j = 0; j < patchHeight; ++j)
        {
            for (int i = 0; i < patchWidth; ++i)
            {
                for (int c = 0; c < channels; ++c)
                {
                    d_simg[(j * patchWidth + i) * channels + c] = d_tmp[(j * patchWidth + (patchWidth - 1 - i)) * channels + c];
                }
            }
        }

        // Step 5: Store the processed patch information in RanderMapPatch
        int patchIdx = y * rawImageParameter.m_xLensNum + x;  // Flattened index
        d_ppRanderMapPatch[patchIdx].sy = curCenterPos.y;
        d_ppRanderMapPatch[patchIdx].sx = curCenterPos.x;
        d_ppRanderMapPatch[patchIdx].simg = d_simg;  // Store the flipped and resized patch
    }
}

void ImageRander::imageRander(float **ppLensMeanDisp, const RawImageParameter &rawImageParameter, 
    const MicroImageParameter &microImageParameter, cv::Mat &randerImg, 
    cv::Mat &destImg)
{
    // 1. 在设备上分配内存
    RanderMapPatch *d_ppRanderMapPatch;
    cudaMalloc(&d_ppRanderMapPatch, rawImageParameter.m_yLensNum * rawImageParameter.m_xLensNum * sizeof(RanderMapPatch));

    // 2. 在主机上获取指向这个内存块的指针
    RanderMapPatch *ppRanderMapPatch = d_ppRanderMapPatch;

    // 3. Allocate device memory for source image, temporary images and render map
    float *d_randerImg, *d_tmp, *d_simg;
    size_t imgSize = randerImg.rows * randerImg.cols * randerImg.channels() * sizeof(float);
    cudaMalloc(&d_randerImg, imgSize);
    cudaMemcpy(d_randerImg, randerImg.ptr<float>(), imgSize, cudaMemcpyHostToDevice);

    size_t tmpSize = DEST_WIDTH * DEST_HEIGHT * randerImg.channels() * sizeof(float);
    cudaMalloc(&d_tmp, tmpSize);
    cudaMalloc(&d_simg, tmpSize);

    // 4. Process each patch (resizing and flipping) in parallel
    dim3 blockSize(16, 16);
    dim3 gridSize((rawImageParameter.m_xLensNum + blockSize.x - 1) / blockSize.x, (rawImageParameter.m_yLensNum + blockSize.y - 1) / blockSize.y);
    processPatchKernel<<<gridSize, blockSize>>>(d_randerImg, d_tmp, d_simg, rawImageParameter, microImageParameter, ppLensMeanDisp, 0, 0, DEST_WIDTH, DEST_HEIGHT, randerImg.channels(), ppRanderMapPatch);
    cudaDeviceSynchronize();

    // 5. Compute boundaries of the render map
    int *sx_begin, *sy_begin, *sx_end, *sy_end;
    cudaMalloc(&sx_begin, sizeof(int));
    cudaMalloc(&sy_begin, sizeof(int));
    cudaMalloc(&sx_end, sizeof(int));
    cudaMalloc(&sy_end, sizeof(int));

    cudaMemset(sx_begin, INT_MAX, sizeof(int));
    cudaMemset(sy_begin, INT_MAX, sizeof(int));
    cudaMemset(sx_end, INT_MIN, sizeof(int));
    cudaMemset(sy_end, INT_MIN, sizeof(int));

    computeBoundaryKernel<<<gridSize, blockSize>>>(ppRanderMapPatch, rawImageParameter, sx_begin, sy_begin, sx_end, sy_end, DEST_WIDTH, DEST_HEIGHT);
    cudaDeviceSynchronize();

    // 6. Retrieve boundary values
    int h_sx_begin, h_sy_begin, h_sx_end, h_sy_end;
    cudaMemcpy(&h_sx_begin, sx_begin, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sy_begin, sy_begin, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sx_end, sx_end, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sy_end, sy_end, sizeof(int), cudaMemcpyDeviceToHost);

    int randerMapWidth = h_sx_end - h_sx_begin + 1;
    int randerMapHeight = h_sy_end - h_sy_begin + 1;

    // 7. Allocate GPU memory for render map and count
    float *d_randerMap, *d_randerCount;
    size_t randerMapSize = randerMapWidth * randerMapHeight * randerImg.channels() * sizeof(float);
    size_t randerCountSize = randerMapWidth * randerMapHeight * sizeof(float);
    cudaMalloc(&d_randerMap, randerMapSize);
    cudaMalloc(&d_randerCount, randerCountSize);
    cudaMemset(d_randerMap, 0, randerMapSize);
    cudaMemset(d_randerCount, 0, randerCountSize);

    // 8. 使用统一的 accumulateKernel 来累加结果
    gridSize.x = (rawImageParameter.m_xLensNum + blockSize.x - 1) / blockSize.x;
    gridSize.y = (rawImageParameter.m_yLensNum + blockSize.y - 1) / blockSize.y;

    // 调用统一的 accumulateKernel 核函数
    accumulateKernel<<<gridSize, blockSize>>>(
        d_randerMap, d_randerCount, d_ppRanderMapPatch, 
        rawImageParameter.m_xLensNum, rawImageParameter.m_yLensNum, 
        DEST_WIDTH, DEST_HEIGHT, randerMapWidth, randerMapHeight, randerImg.channels()
    );
    cudaDeviceSynchronize();

    // 9. Normalize the render map on GPU
    gridSize.x = (randerMapWidth + blockSize.x - 1) / blockSize.x;
    gridSize.y = (randerMapHeight + blockSize.y - 1) / blockSize.y;
    
    // 调用 normalizeKernel 来归一化渲染图
    normalizeKernel<<<gridSize, blockSize>>>(d_randerMap, d_randerCount, randerMapWidth, randerMapHeight, randerImg.channels());

    // 10. Copy the result to host and create output image
    //float *h_randerMap = new float[randerMapWidth * randerMapHeight * randerImg.channels()];
    //cudaMemcpy(h_randerMap, d_randerMap, randerMapSize, cudaMemcpyDeviceToHost);
    //destImg = cv::Mat(randerMapHeight, randerMapWidth, randerImg.type(), h_randerMap);

    // 11. Clean up GPU memory
    cudaFree(d_randerMap);
    cudaFree(d_randerCount);
    cudaFree(d_randerImg);
    cudaFree(d_tmp);
    cudaFree(d_simg);
    cudaFree(d_ppRanderMapPatch);  // Free the allocated memory for the patch data
}



/*__global__ void resizeKernel(float *d_src, float *d_dst, int src_width, int src_height, int dst_width, int dst_height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dst_width && y < dst_height)
    {
        // 计算源图像对应位置的坐标
        float fx = (float)x / (dst_width - 1) * (src_width - 1);
        float fy = (float)y / (dst_height - 1) * (src_height - 1);
        
        int ix = (int)fx;
        int iy = (int)fy;

        // 获取邻域的像素值并执行双线性插值
        float wx = fx - ix;
        float wy = fy - iy;

        for (int c = 0; c < channels; ++c)
        {
            float top_left = d_src[(iy * src_width + ix) * channels + c];
            float top_right = d_src[(iy * src_width + (ix + 1)) * channels + c];
            float bottom_left = d_src[((iy + 1) * src_width + ix) * channels + c];
            float bottom_right = d_src[((iy + 1) * src_width + (ix + 1)) * channels + c];

            float interpolated = (1 - wx) * (1 - wy) * top_left +
                                 wx * (1 - wy) * top_right +
                                 (1 - wx) * wy * bottom_left +
                                 wx * wy * bottom_right;

            d_dst[(y * dst_width + x) * channels + c] = interpolated;
        }
    }
}

void resizeCUDA(float *d_src, float *d_dst, int src_width, int src_height, int dst_width, int dst_height, int channels)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((dst_width + blockSize.x - 1) / blockSize.x, (dst_height + blockSize.y - 1) / blockSize.y);

    resizeKernel<<<gridSize, blockSize>>>(d_src, d_dst, src_width, src_height, dst_width, dst_height, channels);
    cudaDeviceSynchronize();
}

__global__ void flipKernel(float *d_src, float *d_dst, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // 水平翻转：交换左右位置
        int flip_x = width - 1 - x;

        for (int c = 0; c < channels; ++c)
        {
            d_dst[(y * width + x) * channels + c] = d_src[(y * width + flip_x) * channels + c];
        }
    }
}

void flipCUDA(float *d_src, float *d_dst, int width, int height, int channels)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    flipKernel<<<gridSize, blockSize>>>(d_src, d_dst, width, height, channels);
    cudaDeviceSynchronize();
}






void ImageRander::imageRander(float **ppLensMeanDisp, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, cv::Mat &randerImg, cv::Mat &destImg)
{
    RanderMapPatch **ppRanderMapPatch = new RanderMapPatch *[rawImageParameter.m_yLensNum];
    for (int i = 0; i < rawImageParameter.m_yLensNum; i++)
        ppRanderMapPatch[i] = new RanderMapPatch[rawImageParameter.m_xLensNum];

    float *d_randerImg, *d_tmp, *d_simg;
    size_t imgSize = randerImg.rows * randerImg.cols * randerImg.channels() * sizeof(float);
    cudaMalloc(&d_randerImg, imgSize);
    cudaMemcpy(d_randerImg, randerImg.ptr<float>(), imgSize, cudaMemcpyHostToDevice);

    size_t tmpSize = DEST_WIDTH * DEST_HEIGHT * randerImg.channels() * sizeof(float);
    cudaMalloc(&d_tmp, tmpSize);
    cudaMalloc(&d_simg, tmpSize);

    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
    {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
        {
            int blockSize = fabs(std::round(d_ppLensMeanDisp[y * rawImageParameter.m_xLensNum + x]));

            Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];
            int starty = curCenterPos.y - blockSize / 2 - rawImageParameter.m_yPixelBeginOffset;
            int startx = curCenterPos.x - blockSize / 2 - rawImageParameter.m_xPixelBeginOffset;

            float *d_srcImg = d_randerImg + (starty * randerImg.cols + startx) * randerImg.channels();
            resizeCUDA(d_srcImg, d_tmp, blockSize, blockSize, DEST_WIDTH, DEST_HEIGHT, randerImg.channels());
            flipCUDA(d_tmp, d_simg, DEST_WIDTH, DEST_HEIGHT, randerImg.channels());

            ppRanderMapPatch[y][x].sy = curCenterPos.y;
            ppRanderMapPatch[y][x].sx = curCenterPos.x;
            ppRanderMapPatch[y][x].simg = d_simg;
        }
    }

    int sx_begin = INT_MAX, sy_begin = INT_MAX;
    int sx_end = INT_MIN, sy_end = INT_MIN;

    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
    {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
        {
            sy_begin = std::min(sy_begin, ppRanderMapPatch[y][x].sy - DEST_HEIGHT / 2);
            sx_begin = std::min(sx_begin, ppRanderMapPatch[y][x].sx - DEST_WIDTH / 2);
            sy_end = std::max(sy_end, ppRanderMapPatch[y][x].sy + DEST_HEIGHT / 2);
            sx_end = std::max(sx_end, ppRanderMapPatch[y][x].sx + DEST_WIDTH / 2);
        }            
    }

    int randerMapWidth = sx_end - sx_begin + 1;
    int randerMapHeight = sy_end - sy_begin + 1;

    float *d_randerMap, *d_randerCount;
    size_t randerMapSize = randerMapWidth * randerMapHeight * randerImg.channels() * sizeof(float);
    size_t randerCountSize = randerMapWidth * randerMapHeight * sizeof(float);
    cudaMalloc(&d_randerMap, randerMapSize);
    cudaMalloc(&d_randerCount, randerCountSize);
    cudaMemset(d_randerMap, 0, randerMapSize);
    cudaMemset(d_randerCount, 0, randerCountSize);

    dim3 blockSize(16, 16);
    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++)
    {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++)
        {
            int sy_b = ppRanderMapPatch[y][x].sy - DEST_HEIGHT / 2 - sy_begin;
            int sx_b = ppRanderMapPatch[y][x].sx - DEST_WIDTH / 2 - sx_begin;

            dim3 gridSize((DEST_WIDTH + blockSize.x - 1) / blockSize.x, (DEST_HEIGHT + blockSize.y - 1) / blockSize.y);
            accumulateKernel<<<gridSize, blockSize>>>(d_randerMap, d_randerCount, ppRanderMapPatch[y][x].simg, sx_b, sy_b, DEST_WIDTH, DEST_HEIGHT, randerMapWidth, randerMapHeight, randerImg.channels());
        }
    }

    dim3 gridSize((randerMapWidth + blockSize.x - 1) / blockSize.x, (randerMapHeight + blockSize.y - 1) / blockSize.y);
    normalizeKernel<<<gridSize, blockSize>>>(d_randerMap, d_randerCount, randerMapWidth, randerMapHeight, randerImg.channels());

    float *h_randerMap = new float[randerMapWidth * randerMapHeight * randerImg.channels()];
    cudaMemcpy(h_randerMap, d_randerMap, randerMapSize, cudaMemcpyDeviceToHost);

    destImg = cv::Mat(randerMapHeight, randerMapWidth, randerImg.type(), h_randerMap);

    cudaFree(d_randerMap);
    cudaFree(d_randerCount);
    cudaFree(d_randerImg);
    cudaFree(d_tmp);
    cudaFree(d_simg);

    for (int i = 0; i < rawImageParameter.m_yLensNum; i++)
        delete[] ppRanderMapPatch[i];
    delete[] ppRanderMapPatch;
}*/

void ImageRander::outputSparseSceneDepth(string folderName, cv::Mat &sceneSparseDepth, cv::Mat &sceneDepthMask)
{
	std::string storeName;
	ofstream ofs1;
	storeName = folderName + "/sceneInitDisp.txt";
	ofs1.open(storeName, ofstream::out);
	ofstream ofs2;
	storeName = folderName + "/sceneInitDispMask.txt";
	ofs2.open(storeName, ofstream::out);

	for (int y = 0; y < sceneSparseDepth.rows; ++y)
	{
		double *pYSceneDisp = (double *)sceneSparseDepth.ptr<double>(y);
		uchar *pYMask = (uchar *)sceneDepthMask.ptr<uchar>(y);
		for (int x = 0; x < sceneSparseDepth.cols; ++x)
		{
			ofs1 << fixed << setprecision(5) << pYSceneDisp[x] << " ";
			if (pYMask[x] == 255)
				ofs2 << 1 << " ";
			else
				ofs2 << 0 << " ";
			//ofs2 << int(pYMask[x]) << " ";
		}
		ofs1 << endl;
		ofs2 << endl;
	}
	ofs1.close();
	ofs2.close();
}