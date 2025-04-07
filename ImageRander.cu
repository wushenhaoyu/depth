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
__global__ void computeLensMeanDispKernel(MicroImageParameterDevice* d_microImageParameter,float* d_rawDisp_temp)
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
                    sum += d_rawDisp_temp[globalY * d_rawImageParameter.m_recImgWidth + globalX];
                    count++;
                }
            }
        }

        float meanDisp = sum / count;
        d_ppLensMeanDisp[y * d_rawImageParameter.m_xLensNum + x] = fmax(meanDisp, (float)d_disparityParameter.m_dispMin);
         
    }
}

__global__ void transRawDisp(float* d_rawDisp,float* d_rawDisp_temp)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(y <= d_rawImageParameter.m_recImgHeight || x <= d_rawImageParameter.m_recImgWidth)
     {
        float a = d_disparityParameter.m_dispStep;
        float b = d_disparityParameter.m_dispMin;
        d_rawDisp_temp[y * d_rawImageParameter.m_recImgWidth + x] = d_rawDisp[y * d_rawImageParameter.m_recImgWidth + x] * a + b;
     }
    
}

__global__ void printLensCenterPoints(MicroImageParameterDevice* d_microImageParameterDevice, int yLensNum, int xLensNum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < yLensNum * xLensNum) {
        int y = idx / xLensNum;
        int x = idx % xLensNum;

        // 获取当前点的坐标
        CudaPoint2f point = d_microImageParameterDevice->m_ppLensCenterPoints[idx];
        printf("x:%d,y:%d,px:%d,py:%d\n",x,y,point.x,point.y);

    }
}


void ImageRander::imageRanderWithOutMask(const DataParameter &dataParameter)
{
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();



        // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((rawImageParameter.m_recImgWidth + blockSize.x - 1) / blockSize.x, 
                      (rawImageParameter.m_recImgHeight + blockSize.y - 1) / blockSize.y);

    /*transRawDisp<<<gridSize, blockSize>>>(d_rawDisp, d_rawDisp_temp);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());*/

    dim3 block(16);
    dim3 grid((rawImageParameter.m_xLensNum * rawImageParameter.m_yLensNum + block.x - 1) / block.x);
    // Launch the kernel to print lens center points
    printLensCenterPoints<<<grid, block>>>(d_microImageParameter, rawImageParameter.m_yLensNum, rawImageParameter.m_xLensNum);
    // Check for any errors during kernel launch
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());



        // Define block and grid sizes
        /*dim3 blockSize(32, 32);
        dim3 gridSize((rawImageParameter.m_xLensNum + blockSize.x - 1) / blockSize.x, 
                      (rawImageParameter.m_yLensNum + blockSize.y - 1) / blockSize.y);*/
        blockSize = dim3(32, 32);
        gridSize = dim3((rawImageParameter.m_xLensNum + blockSize.x - 1) / blockSize.x, 
                      (rawImageParameter.m_yLensNum + blockSize.y - 1) / blockSize.y);

    computeLensMeanDispKernel<<<gridSize, blockSize>>>(d_microImageParameter,d_rawDisp_temp);

    // Check for any errors during kernel launch
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());



	imageRander(rawImageParameter, microImageParameter,d_inputImgRec,3);
    saveThreeChannelGpuMemoryAsImage(d_randerMap,  randerMapHeightVal,randerMapWidthVal_, "result_3.bmp");
    imageRander(rawImageParameter, microImageParameter,d_rawDisp,1);
    saveSingleChannelGpuMemoryAsImage(d_randerMap, randerMapHeightVal,randerMapWidthVal_, "result_1.bmp");

}


__global__ void accumulateKernel(RanderMapPatch* d_ppRanderMapPatch,float* d_randerMap,float* d_randerCount,
    int DEST_WIDTH_, 
    int DEST_HEIGHT_, 
    int channels,int* sy_begin,int* sx_begin)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 获取 randerMapWidth 和 randerMapHeight 通过解引用指针
    int randerMapWidthVal = d_randerMapWidth[0];
    int randerMapHeightVal = d_randerMapHeight[0];

    if (x < d_rawImageParameter.m_xLensNum && y < d_rawImageParameter.m_yLensNum)
    {
        // 使用线性索引来访问 RanderMapPatch
        RanderMapPatch patch = d_ppRanderMapPatch[y * d_rawImageParameter.m_xLensNum + x];

        // 计算在输出图像中的起始点
        int sy_b = patch.sy - DEST_HEIGHT_ / 2 - sy_begin[0];
        int sx_b = patch.sx - DEST_WIDTH_ / 2 - sx_begin[0];

        // 处理补丁
        for (int py = 0; py < DEST_HEIGHT_; ++py)
        {
            for (int px = 0; px < DEST_WIDTH_; ++px)
            {
                int rander_x = sx_b + px;
                int rander_y = sy_b + py;


                if (rander_x >= 0 && rander_x < randerMapWidthVal && rander_y >= 0 && rander_y < randerMapHeightVal)
                {
                    // 将补丁添加到渲染图
                    for (int c = 0; c < channels; ++c)
                    {
                        atomicAdd(&d_randerMap[(rander_y * randerMapWidthVal + rander_x) * channels + c], patch.simg[(py * DEST_WIDTH_ + px) * channels + c]);
                    }

                    // 统计每个像素被多少个补丁贡献
                    atomicAdd(&d_randerCount[rander_y * randerMapWidthVal + rander_x], 1.0f);
                }
            }
        }
    }
}


__global__ void normalizeKernel(float* d_randerMap,float* d_randerCount,int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 获取宽度和高度，通过解引用指针
    int randerMapWidth = d_randerMapWidth[0];
    int randerMapHeight = d_randerMapHeight[0];

    if (x < randerMapWidth && y < randerMapHeight)
    {
        for (int c = 0; c < channels; ++c)
        {
            int idx = (y * randerMapWidth + x) * channels + c;
            if (d_randerCount[y * randerMapWidth + x] > 0)
            {
                d_randerMap[idx] /= d_randerCount[y * randerMapWidth + x];
            }
        }
    }
}


__global__ void computeBoundaryKernel(RanderMapPatch* d_ppRanderMapPatch,
    int DEST_WIDTH_, int DEST_HEIGHT_,int* d_sx_begin, int* d_sy_begin, int* d_sx_end, int* d_sy_end)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x < d_rawImageParameter.m_xLensNum && y < d_rawImageParameter.m_yLensNum &&
        x >= 0 && y >= 0) // Ensure x and y are within valid bounds
    {
        // 使用一维数组访问
        int sy = d_ppRanderMapPatch[y * d_rawImageParameter.m_xLensNum + x].sy;
        int sx = d_ppRanderMapPatch[y * d_rawImageParameter.m_xLensNum + x].sx;
        printf("x:%d y:%d sx:%d sy:%d\n",x,y,sx,sy);
        atomicMin(d_sx_begin, sx - DEST_WIDTH_ / 2);
        atomicMin(d_sy_begin, sy - DEST_HEIGHT_ / 2);
        atomicMax(d_sx_end, sx + DEST_WIDTH_ / 2);
        atomicMax(d_sy_end, sy + DEST_HEIGHT_ / 2);
    }
}


__global__ void processPatchKernel(MicroImageParameterDevice* d_microImageParameter,RanderMapPatch* d_ppRanderMapPatch, float* d_input,
    int patchWidth, int patchHeight,int Channels) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = threadIdx.z;  // 处理的颜色通道 (0, 1, 2)

    // Adjust the range to match the CPU code
    int xAdjusted = x + d_rawImageParameter.m_xCenterBeginOffset;
    int yAdjusted = y + d_rawImageParameter.m_yCenterBeginOffset;

    if (xAdjusted >= d_rawImageParameter.m_xLensNum - d_rawImageParameter.m_xCenterEndOffset ||
        yAdjusted >= d_rawImageParameter.m_yLensNum - d_rawImageParameter.m_yCenterEndOffset) {
        return;
    }

    CudaPoint2f curCenterPos = d_microImageParameter->m_ppLensCenterPoints[y * d_rawImageParameter.m_xLensNum + x];
    int blockSize = fabsf(roundf(d_ppLensMeanDisp[yAdjusted * d_rawImageParameter.m_xLensNum + xAdjusted]));
    int starty = max(static_cast<int>(curCenterPos.y - blockSize / 2 - d_rawImageParameter.m_yPixelBeginOffset), 0);
    int startx = max(static_cast<int>(curCenterPos.x - blockSize / 2 - d_rawImageParameter.m_xPixelBeginOffset), 0);

    d_ppRanderMapPatch[yAdjusted * d_rawImageParameter.m_xLensNum + xAdjusted].sy = int(curCenterPos.y);
    d_ppRanderMapPatch[yAdjusted * d_rawImageParameter.m_xLensNum + xAdjusted].sx = int(curCenterPos.x);
    float *d_srcImg = d_input + (starty * d_rawImageParameter.m_xLensNum + startx) * Channels;
    float *d_simg = d_ppRanderMapPatch[yAdjusted * d_rawImageParameter.m_xLensNum + xAdjusted].simg;
    printf("x:%d y:%d,rx:%d,ry:%d, sx:%d sy:%d\n",xAdjusted,yAdjusted,curCenterPos.x,curCenterPos.y,d_ppRanderMapPatch[yAdjusted * d_rawImageParameter.m_xLensNum + xAdjusted].sx,d_ppRanderMapPatch[yAdjusted * d_rawImageParameter.m_xLensNum + xAdjusted].sy);

    int i = threadIdx.x;
    int j = threadIdx.y;

if (i < patchWidth && j < patchHeight) {
    // 计算双线性插值
    float fx = (float)i / (patchWidth - 1) * (blockSize - 1);
    float fy = (float)j / (patchHeight - 1) * (blockSize - 1);
    int ix = (int)fx;
    int iy = (int)fy;
    float wx = fx - ix;
    float wy = fy - iy;

    float top_left = d_srcImg[(iy * blockSize + ix) * Channels + c];
    float top_right = d_srcImg[(iy * blockSize + ix + 1) * Channels + c];
    float bottom_left = d_srcImg[((iy + 1) * blockSize + ix) * Channels + c];
    float bottom_right = d_srcImg[((iy + 1) * blockSize + ix + 1) * Channels + c];

    float interpolated = (1 - wx) * (1 - wy) * top_left +
                        wx * (1 - wy) * top_right +
                        (1 - wx) * wy * bottom_left +
                        wx * wy * bottom_right;

    d_simg[(j * patchWidth + i) * Channels + c] = interpolated;

    // 镜像翻转 Patch
    d_simg[(j * patchWidth + (patchWidth - 1 - i)) * Channels + c] = interpolated;

    // 存储 Patch 位置
        /*if (c == 0 && i == 0 && j == 0) {
            int patchIdx = yAdjusted * d_rawImageParameter.m_xLensNum + xAdjusted;
            d_ppRanderMapPatch[patchIdx].sy = curCenterPos.y;
            d_ppRanderMapPatch[patchIdx].sx = curCenterPos.x;
        }*/
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
    dim3 blockSize(16, 16, Channels);  // (Patch 16x16, 每个线程负责一个像素，3 个通道)
    dim3 gridSize((rawImageParameter.m_xLensNum + blockSize.x - 1) / blockSize.x, 
                  (rawImageParameter.m_yLensNum + blockSize.y - 1) / blockSize.y);
    
    processPatchKernel<<<gridSize, blockSize>>>(d_microImageParameter,d_ppRanderMapPatch,d_input,DEST_WIDTH, DEST_HEIGHT,Channels);
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
    
    
        h_randerMapWidth = h_sx_end - h_sx_begin + 1;
        h_randerMapHeight = h_sy_end - h_sy_begin + 1;

        randerMapHeightVal = h_randerMapHeight;
        randerMapWidthVal = h_randerMapWidth;
        randerMapWidthVal_ = h_randerMapWidth; //很蠢的cuda,不知道为啥会改变randerMapWidthVal
        CUDA_CHECK(cudaMalloc((void**)&h_randerMapWidth, sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_randerMapWidth, &h_randerMapWidth, sizeof(int*)));

        CUDA_CHECK(cudaMalloc((void**)&h_randerMapHeight, sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_randerMapHeight, &h_randerMapHeight, sizeof(int*)));
    
    
        size_t randerMapSize = randerMapWidthVal_ * randerMapHeightVal * Channels * sizeof(float); 
        size_t randerCountSize = randerMapWidthVal_  * randerMapHeightVal * sizeof(float);
    
    
        CUDA_CHECK(cudaMalloc(&d_randerMap, randerMapSize));
        CUDA_CHECK(cudaMalloc(&d_randerCount, randerCountSize));
    
        // 初始化设备内存
        CUDA_CHECK(cudaMemset(d_randerMap, 0, randerMapSize));
        CUDA_CHECK(cudaMemset(d_randerCount, 0, randerCountSize));
        isCalWH = 1;
    }




    // Step 5: Accumulate kernel
    cudaEventRecord(start); // 记录开始时间
    gridSize.x = (rawImageParameter.m_xLensNum + blockSize.x - 1) / blockSize.x;
    gridSize.y = (rawImageParameter.m_yLensNum + blockSize.y - 1) / blockSize.y;

    accumulateKernel<<<gridSize, blockSize>>>(d_ppRanderMapPatch,d_randerMap,d_randerCount,DEST_WIDTH, DEST_HEIGHT, Channels,d_sy_begin,d_sx_begin); // 3通道
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&ms, start, stop);
    printf("Accumulate Kernel Time: %f ms\n", ms); 


    // Step 6: Normalize kernel
    cudaEventRecord(start); 
    gridSize.x = ( randerMapWidthVal_ + blockSize.x - 1) / blockSize.x;
    gridSize.y = (randerMapHeightVal + blockSize.y - 1) / blockSize.y;
    normalizeKernel<<<gridSize, blockSize>>>(d_randerMap,d_randerCount,Channels); // 3通道
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&ms, start, stop);
    printf("Normalize Kernel Time: %f ms\n", ms); 

    /*还差去除边界空洞*/


   
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}








/*
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