#include "ImageRander.h"
#include "DataParameter.cuh"
#include <iomanip>

using namespace std;
using namespace cv;



ImageRander::ImageRander()
{

}

ImageRander::~ImageRander()
{

}


int isCalWH = 0;
 float *d_randerMap, *d_randerCount;
 int h_randerMapWidth , h_randerMapHeight;
 int randerMapWidthVal, randerMapHeightVal;
 int randerMapWidthVal_;
 int x_left, x_right, y_top, y_below;
__global__ void computeLensMeanDispKernel(MicroImageParameterDevice* d_microImageParameter,float* d_rawDisp)
{
    // 获取当前线程的坐标
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保线程在有效范围内
    if (y >= d_rawImageParameter.m_yCenterBeginOffset && y < d_rawImageParameter.m_yLensNum - d_rawImageParameter.m_yCenterEndOffset &&
        x >= d_rawImageParameter.m_xCenterBeginOffset && x < d_rawImageParameter.m_xLensNum - d_rawImageParameter.m_xCenterEndOffset)
    {
        // 计算当前中心点坐标
        CudaPoint2f curCenterPos = CudaPoint2f(d_microImageParameter->m_ppLensCenterPoints[y * d_rawImageParameter.m_xLensNum + x].x, d_microImageParameter->m_ppLensCenterPoints[y * d_rawImageParameter.m_xLensNum + x].y);
       // printf("x:%d y:%d sx:%f sy:%f\n",x,y,curCenterPos.x,curCenterPos.y);
        int x_begin = curCenterPos.x - d_rawImageParameter.m_xPixelBeginOffset - d_meanDispLenRadius;
        int y_begin = curCenterPos.y - d_rawImageParameter.m_yPixelBeginOffset - d_meanDispLenRadius;
        // 计算区域的宽度和高度
        int rectWidth = d_meanDispLenRadius * 2 + 1;
        int rectHeight = d_meanDispLenRadius * 2 + 1;

        // 计算该区域的均值
        float sum = 0.0f;
        int count = 0;

        for (int dy = 0; dy < rectHeight; dy++) {
            for (int dx = 0; dx < rectWidth; dx++) {
                int globalX = x_begin + dx;
                int globalY = y_begin + dy;

                // 确保访问的坐标在有效范围内
                if (globalX >= 0 && globalX < d_rawImageParameter.m_recImgWidth &&
                    globalY >= 0 && globalY < d_rawImageParameter.m_recImgHeight)
                {
                    sum += d_rawDisp[globalY * d_rawImageParameter.m_recImgWidth + globalX] *255 * d_disparityParameter.m_dispStep + d_disparityParameter.m_dispMin;
                    count++;
                }
            }
        }

        float meanDisp = sum / count ;
        d_ppLensMeanDisp[y * d_rawImageParameter.m_xLensNum + x] = fmax(meanDisp, 9.0f);

    }
}





void ImageRander::imageRanderWithOutMask(const DataParameter &dataParameter)
{
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Step 1: Process patch kernel
    cudaEventRecord(start);

        // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((rawImageParameter.m_recImgWidth + blockSize.x - 1) / blockSize.x, 
                      (rawImageParameter.m_recImgHeight + blockSize.y - 1) / blockSize.y);

        blockSize = dim3(32, 32);
        gridSize = dim3((rawImageParameter.m_xLensNum + blockSize.x - 1) / blockSize.x, 
                      (rawImageParameter.m_yLensNum + blockSize.y - 1) / blockSize.y);

    computeLensMeanDispKernel<<<gridSize, blockSize>>>(d_microImageParameter,d_rawDisp);

    // Check for any errors during kernel launch
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Compute Lens Mean Disp Kernel Time: %f ms\n", ms);

    



    cudaMemset(d_randerMap, 0, randerMapWidthVal_ * randerMapHeightVal * 3 * sizeof(float));
    imageRander(rawImageParameter, microImageParameter,d_inputImgRec,3);
    saveThreeChannelGpuMemoryAsImage(d_randerMap,  randerMapWidthVal_,randerMapHeightVal, "./res/randerSceneMap.bmp");
    cudaMemset(d_randerMap, 0, randerMapWidthVal_ * randerMapHeightVal * 3 * sizeof(float));
    imageRander(rawImageParameter, microImageParameter,d_rawDisp,1);
    saveSingleChannelGpuMemoryAsImage(d_randerMap, randerMapWidthVal_,randerMapHeightVal, "./res/randerDisMap.bmp");
    //保存图像会耗费约40ms时间
}


__global__ void accumulateKernel(
    RanderMapPatch* d_ppRanderMapPatch,
    float* d_randerMap,
    float* d_randerCount,
    int patchWidth,
    int patchHeight,
    int channels,
    int* sy_begin,
    int* sx_begin,
    int randerMapWidth,
    int randerMapHeight)
{
    // 当前 patch 坐标
    int patchX = blockIdx.x;
    int patchY = blockIdx.y;

    if (patchX >= d_rawImageParameter.m_xLensNum || patchY >= d_rawImageParameter.m_yLensNum)
        return;

    RanderMapPatch patch = d_ppRanderMapPatch[patchY * d_rawImageParameter.m_xLensNum + patchX];

    int sy_b = patch.sy - patchHeight / 2 - sy_begin[0];
    int sx_b = patch.sx - patchWidth  / 2 - sx_begin[0];

    // 当前线程的处理起始坐标
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int c = threadIdx.z;

    int stride_x = blockDim.x;
    int stride_y = blockDim.y;


    if (c >= channels)
        return;

    for (int py = ty; py < patchHeight; py += stride_y)
    {
        for (int px = tx; px < patchWidth; px += stride_x)
        {

            int rander_x = sx_b + px;
            int rander_y = sy_b + py;

            if (rander_x >= 0 && rander_x < randerMapWidth &&
                rander_y >= 0 && rander_y < randerMapHeight)
            {
                int simgIdx = (py * patchWidth + px) * channels + c;
                int dstIdx  = (rander_y * randerMapWidth + rander_x) * channels + c;

                float val = patch.simg[simgIdx];

                atomicAdd(&d_randerMap[dstIdx], val);

                if (c == 0)
                    atomicAdd(&d_randerCount[rander_y * randerMapWidth + rander_x], 1.0f);
            }
        }
    }
}

__global__ void normalizeKernel(float* d_randerMap,float* d_randerCount,int channels,int d_randerMapWidth,int d_randerMapHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x < d_randerMapWidth && y < d_randerMapHeight)
    {
        for (int c = 0; c < channels; ++c)
        {
            int idx = (y * d_randerMapWidth + x) * channels + c;
            if (d_randerCount[y * d_randerMapWidth + x] > 0)
            {
                d_randerMap[idx] /= d_randerCount[y * d_randerMapWidth + x];
            }
        }
        
    }
}


__global__ void computeBoundaryKernel(RanderMapPatch* d_ppRanderMapPatch,
    int DEST_WIDTH_, int DEST_HEIGHT_,int* d_sx_begin, int* d_sy_begin, int* d_sx_end, int* d_sy_end)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x < d_rawImageParameter.m_xLensNum - d_rawImageParameter.m_xCenterEndOffset && y < d_rawImageParameter.m_yLensNum  - d_rawImageParameter.m_yCenterEndOffset &&
        x >= d_rawImageParameter.m_xCenterBeginOffset && y >= d_rawImageParameter.m_yCenterBeginOffset) // Ensure x and y are within valid bounds
    {
        // 使用一维数组访问
        int sy = d_ppRanderMapPatch[y * d_rawImageParameter.m_xLensNum + x].sy;
        int sx = d_ppRanderMapPatch[y * d_rawImageParameter.m_xLensNum + x].sx;
       // printf("x:%d y:%d sx:%d sy:%d\n",x,y,sx,sy);
        atomicMin(d_sx_begin, sx - DEST_WIDTH_ / 2);
        atomicMin(d_sy_begin, sy - DEST_HEIGHT_ / 2);
        atomicMax(d_sx_end, sx + DEST_WIDTH_ / 2);
        atomicMax(d_sy_end, sy + DEST_HEIGHT_ / 2);
    }
}


__global__ void processPatchKernel(MicroImageParameterDevice* d_microImageParameter, 
    RanderMapPatch* d_ppRanderMapPatch, 
    float* d_input,
    int patchWidth, int patchHeight, int Channels)
{
    // 当前处理的 patch 位置（一个线程块处理一个 patch）
    int patchX = blockIdx.x;
    int patchY = blockIdx.y;

    int c = threadIdx.z;

    // 每个线程从哪个像素开始
    int i_start = threadIdx.x;
    int j_start = threadIdx.y;

    // 每个线程跨步处理多个像素
    int stride_i = blockDim.x;
    int stride_j = blockDim.y;

    if (c >= Channels) return;

    // 获取 patch 相关信息
    int xAdjusted = patchX + d_rawImageParameter.m_xCenterBeginOffset;
    int yAdjusted = patchY + d_rawImageParameter.m_yCenterBeginOffset;

    if (xAdjusted >= d_rawImageParameter.m_xLensNum - d_rawImageParameter.m_xCenterEndOffset ||
    yAdjusted >= d_rawImageParameter.m_yLensNum - d_rawImageParameter.m_yCenterEndOffset)
    return;

    CudaPoint2f curCenterPos = d_microImageParameter->m_ppLensCenterPoints[yAdjusted * d_rawImageParameter.m_xLensNum + xAdjusted];
    int blockSize = fabsf(roundf(d_ppLensMeanDisp[yAdjusted * d_rawImageParameter.m_xLensNum + xAdjusted]));

    int starty = max(static_cast<int>(curCenterPos.y - blockSize / 2 - d_rawImageParameter.m_yPixelBeginOffset), 0);
    int startx = max(static_cast<int>(curCenterPos.x - blockSize / 2 - d_rawImageParameter.m_xPixelBeginOffset), 0);

    d_ppRanderMapPatch[yAdjusted * d_rawImageParameter.m_xLensNum + xAdjusted].sy = int(curCenterPos.y);
    d_ppRanderMapPatch[yAdjusted * d_rawImageParameter.m_xLensNum + xAdjusted].sx = int(curCenterPos.x);

    float* d_srcImg = d_input + (starty * d_rawImageParameter.m_recImgWidth + startx) * Channels;
    float* d_simg = d_ppRanderMapPatch[yAdjusted * d_rawImageParameter.m_xLensNum + xAdjusted].simg;




    int imageStride = d_rawImageParameter.m_recImgWidth;  // 原图宽度

    for (int j = j_start; j < patchHeight; j += stride_j) {
        for (int i = i_start; i < patchWidth; i += stride_i) {
            float fx = (float)i / (patchWidth - 1) * (blockSize - 1);
            float fy = (float)j / (patchHeight - 1) * (blockSize - 1);
            int ix = (int)fx;
            int iy = (int)fy;
            float wx = fx - ix;
            float wy = fy - iy;
    
            // 全局坐标（以原图为基准）
            int global_x = startx + ix;
            int global_y = starty + iy;
    
            // 插值使用原图内存访问
            float top_left     = d_input[(global_y * imageStride + global_x) * Channels + c];
            float top_right    = d_input[(global_y * imageStride + global_x + 1) * Channels + c];
            float bottom_left  = d_input[((global_y + 1) * imageStride + global_x) * Channels + c];
            float bottom_right = d_input[((global_y + 1) * imageStride + global_x + 1) * Channels + c];
    
            float interpolated = (1 - wx) * (1 - wy) * top_left +
                                 wx * (1 - wy) * top_right +
                                 (1 - wx) * wy * bottom_left +
                                 wx * wy * bottom_right;
    
            // 写入输出 patch（局部 patch 图像）
            //d_simg[(j * patchWidth + i) * Channels + c] = interpolated;
            int flip_x = patchWidth - i - 1;  // 水平翻转
            int flip_y = patchHeight - j - 1; // 垂直翻转

            d_simg[(flip_y * patchWidth + flip_x) * Channels + c] = interpolated;
        }
    }
    

}


__global__ void repairImageKernel(
    const RawImageParameter d_rawImageParameter,
    RanderMapPatch* patches,
    int* sx_begin, int* sy_begin,
    int* x_left, int* x_right,
    int* y_top, int* y_below,
    int patch_width, int patch_height)
{
// 计算当前线程的x、y坐标
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

    // 计算有效的边界范围
    int left   = d_rawImageParameter.m_xCenterBeginOffset;
    int right  = d_rawImageParameter.m_xLensNum - d_rawImageParameter.m_xCenterEndOffset - 1;
    int top    = d_rawImageParameter.m_yCenterBeginOffset;
    int below  = d_rawImageParameter.m_yLensNum - d_rawImageParameter.m_yCenterEndOffset - 1;

    // 判断当前线程是否在有效区域内，如果在，则进行计算
    if (x >= left && x <= right && y >= top && y <= below)
    {
    int index = y * d_rawImageParameter.m_xLensNum + x;
    int sx = patches[index].sx;
    int sy = patches[index].sy;

    // 计算patch的边界
    int patch_x_left   = sx - patch_width / 2 - sx_begin[0];
    int patch_x_right  = sx + patch_width / 2 - sx_begin[0];
    int patch_y_top    = sy - patch_height / 2 - sy_begin[0];
    int patch_y_below  = sy + patch_height / 2 - sy_begin[0];

    // 使用原子操作来更新全局最小/最大值
    atomicMin(x_left, patch_x_left);
    atomicMax(x_right, patch_x_right);
    atomicMin(y_top, patch_y_top);
    atomicMax(y_below, patch_y_below);
    }
}








void ImageRander::imageRander(const RawImageParameter &rawImageParameter, 
    const MicroImageParameter &microImageParameter,float* d_input,int Channels)
{
   
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Step 1: Process patch kernel
    cudaEventRecord(start); 
    dim3 blockSize(16, 16, Channels);  

    // 图像中 patch 的总数量（X × Y）
    dim3 gridSize(
        rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterBeginOffset - rawImageParameter.m_xCenterEndOffset,
        rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterBeginOffset - rawImageParameter.m_yCenterEndOffset
    );
    
    // 启动 kernel
    processPatchKernel<<<gridSize, blockSize>>>(
        d_microImageParameter,
        d_ppRanderMapPatch,
        d_input,
        DEST_WIDTH,
        DEST_HEIGHT,
        Channels
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Process Patch Kernel Time: %f ms\n", ms); 

    if(!isCalWH) /*只计算一次就够*/
    {
        int h_sx_begin = INT_MAX, h_sy_begin = INT_MAX, h_sx_end = INT_MIN, h_sy_end = INT_MIN;
        CUDA_CHECK(cudaMemcpy(d_sx_begin, &h_sx_begin, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sy_begin, &h_sy_begin, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sx_end, &h_sx_end, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sy_end, &h_sy_end, sizeof(int), cudaMemcpyHostToDevice));
    
        // Step 2: Compute boundary kernel
        cudaEventRecord(start); 
        blockSize = dim3(32, 32);
        gridSize = dim3((rawImageParameter.m_xLensNum + blockSize.x - 1) / blockSize.x, (rawImageParameter.m_yLensNum + blockSize.y - 1) / blockSize.y);
        computeBoundaryKernel<<<gridSize, blockSize>>>(d_ppRanderMapPatch,DEST_WIDTH, DEST_HEIGHT,d_sx_begin,d_sy_begin,d_sx_end,d_sy_end);
        CUDA_CHECK(cudaGetLastError()); 
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(stop); 
        cudaEventSynchronize(stop); 
        cudaEventElapsedTime(&ms, start, stop);
        printf("Compute Boundary Kernel Time: %f ms\n", ms); 
    
    
        // Step 3: Compute width and height kernel
    
        CUDA_CHECK(cudaMemcpy(&h_sx_begin, d_sx_begin, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_sy_begin, d_sy_begin, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_sx_end, d_sx_end, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_sy_end, d_sy_end, sizeof(int), cudaMemcpyDeviceToHost));
    
        printf("sy_begin:%d sy_end:%d sx_begin:%d sx_end:%d\n",h_sy_begin,h_sy_end,h_sx_begin,h_sx_end);
        h_randerMapWidth = h_sx_end - h_sx_begin + 1;
        h_randerMapHeight = h_sy_end - h_sy_begin + 1;

        randerMapHeightVal = h_randerMapHeight;
        randerMapWidthVal = h_randerMapWidth;
        randerMapWidthVal_ = h_randerMapWidth; 
        printf("randerMapWidth:%d randerMapHeight:%d\n",h_randerMapWidth,h_randerMapHeight);
        //1011 1866
    
        size_t randerMapSize = randerMapWidthVal_ * randerMapHeightVal * Channels * sizeof(float); 
        size_t randerCountSize = randerMapWidthVal_  * randerMapHeightVal * sizeof(float);
    
    
        CUDA_CHECK(cudaMalloc(&d_randerMap, randerMapSize));
        CUDA_CHECK(cudaMalloc(&d_randerCount, randerCountSize));
    
        // 初始化设备内存
        CUDA_CHECK(cudaMemset(d_randerMap, 0, randerMapSize));
        CUDA_CHECK(cudaMemset(d_randerCount, 0, randerCountSize));
        
    }




    // Step 5: Accumulate kernel
    cudaEventRecord(start); // 记录开始时间
    blockSize = dim3(16, 16, Channels); 
    gridSize=  dim3(rawImageParameter.m_xLensNum, rawImageParameter.m_yLensNum); 

    accumulateKernel<<<gridSize, blockSize>>>(d_ppRanderMapPatch,d_randerMap,d_randerCount,DEST_WIDTH, DEST_HEIGHT, Channels,d_sy_begin,d_sx_begin,randerMapWidthVal_,randerMapHeightVal); // 3通道
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&ms, start, stop);
    printf("Accumulate Kernel Time: %f ms\n", ms); 


    // Step 6: Normalize kernel
    cudaEventRecord(start); 
    blockSize = dim3(32, 32);
    gridSize.x = ( randerMapWidthVal_ + blockSize.x - 1) / blockSize.x;
    gridSize.y = (randerMapHeightVal + blockSize.y - 1) / blockSize.y;
    normalizeKernel<<<gridSize, blockSize>>>(d_randerMap,d_randerCount,Channels,randerMapWidthVal_,randerMapHeightVal); // 3通道
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&ms, start, stop);
    printf("Normalize Kernel Time: %f ms\n", ms); 

    if(!isCalWH)
    {
        int* d_x_left;
        int* d_x_right;
        int* d_y_top;
        int* d_y_below;
        cudaMalloc(&d_x_left, sizeof(int));
        cudaMalloc(&d_x_right, sizeof(int));
        cudaMalloc(&d_y_top, sizeof(int));
        cudaMalloc(&d_y_below, sizeof(int));

        // 初始化边界值
        int initValue = 0;
        cudaMemcpy(d_x_left, &initValue, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_right, &randerMapWidthVal_, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_top, &initValue, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_below, &randerMapHeightVal, sizeof(int), cudaMemcpyHostToDevice);
        blockSize = dim3(32, 32);  // 这里可以根据你的硬件调整
        gridSize = dim3((rawImageParameter.m_xLensNum + blockSize.x - 1) / blockSize.x,
                      (rawImageParameter.m_yLensNum + blockSize.y - 1) / blockSize.y);
        
        repairImageKernel<<<gridSize, blockSize>>>(
            d_rawImageParameter,
            d_ppRanderMapPatch,
            d_sx_begin, d_sy_begin,
            d_x_left, d_x_right,
            d_y_top, d_y_below,
            DEST_WIDTH, DEST_HEIGHT
        );

        CUDA_CHECK(cudaGetLastError()); 
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaMemcpy(&x_left, d_x_left, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&x_right, d_x_right, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&y_top, d_y_top, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&y_below, d_y_below, sizeof(int), cudaMemcpyDeviceToHost);
        isCalWH = 1 ;
    }


   
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


void saveSingleChannelGpuMemoryAsImage(float* d_data, int width, int height, const std::string& filename) {
    // 分配主机内存
    float* h_data = new float[width * height];

    // 从设备内存拷贝到主机内存
    CUDA_CHECK(cudaMemcpy(h_data, d_data, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    // 将浮点数据转换为 uchar 数据
    cv::Mat img(height, width, CV_32FC1, h_data);
	double minVal; double maxVal;
	minMaxLoc(img, &minVal, &maxVal);
    cv::Mat img_8u;
    img(cv::Rect(x_left, y_top, x_right - x_left, y_below - y_top)).convertTo(img_8u, CV_8UC1,255.0 / (maxVal - minVal), -minVal*255.0 / (maxVal - minVal)); // 将浮点值 [0, 1] 转换为 [0, 255]

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
    img(cv::Rect(x_left, y_top, x_right - x_left, y_below - y_top)).convertTo(img_8u, CV_8UC3, 255.0); // 将浮点值 [0, 1] 转换为 [0, 255]

    // 保存图像
    cv::imwrite(filename, img_8u);

    // 释放主机内存
    delete[] h_data;
}


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