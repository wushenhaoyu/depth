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
        //if(x==20&&y==20)
        //{
            //printf("x:%d y:%d meanDisp:%f\n",x,y,meanDisp);
        //}
        //printf("x:%d y:%d meanDisp:%f\n",x,y,meanDisp);
        /*if(x==21&&y==48)
        {
            printf("x:%d y:%d meanDisp:%f\n",x,y,meanDisp);
            printf("x_begin:%d y_begin:%d\n,",x_begin,y_begin);
            printf("recImgWidth:%d recImgHeight:%d\n",rectWidth,rectHeight);
        }*/
    }
}



__global__ void printLensCenterPoints(MicroImageParameterDevice* d_microImageParameterDevice, int yLensNum, int xLensNum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < yLensNum * xLensNum) {
        int y = idx / xLensNum;
        int x = idx % xLensNum;

        // 获取当前点的坐标
        CudaPoint2f point = d_microImageParameterDevice->m_ppLensCenterPoints[idx];
        //printf("x:%d,y:%d,px:%d,py:%d\n",x,y,point.x,point.y);

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

   /* dim3 block(16);
    dim3 grid((rawImageParameter.m_xLensNum * rawImageParameter.m_yLensNum + block.x - 1) / block.x);
    // Launch the kernel to print lens center points
    printLensCenterPoints<<<grid, block>>>(d_microImageParameter, rawImageParameter.m_yLensNum, rawImageParameter.m_xLensNum);
    // Check for any errors during kernel launch
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());*/



        // Define block and grid sizes
        /*dim3 blockSize(32, 32);
        dim3 gridSize((rawImageParameter.m_xLensNum + blockSize.x - 1) / blockSize.x, 
                      (rawImageParameter.m_yLensNum + blockSize.y - 1) / blockSize.y);*/
        blockSize = dim3(32, 32);
        gridSize = dim3((rawImageParameter.m_xLensNum + blockSize.x - 1) / blockSize.x, 
                      (rawImageParameter.m_yLensNum + blockSize.y - 1) / blockSize.y);

    computeLensMeanDispKernel<<<gridSize, blockSize>>>(d_microImageParameter,d_rawDisp);

    // Check for any errors during kernel launch
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());



	//imageRander(rawImageParameter, microImageParameter,d_inputImgRec,3);
    //saveThreeChannelGpuMemoryAsImage(d_randerMap,  randerMapWidthVal_,randerMapHeightVal, "./res/randerSceneMap.bmp");
    imageRander(rawImageParameter, microImageParameter,d_rawDisp,1);
    saveSingleChannelGpuMemoryAsImage(d_randerMap, randerMapWidthVal_,randerMapHeightVal, "./res/randerDisMap.bmp");

}


__global__ void accumulateKernel(RanderMapPatch* d_ppRanderMapPatch,float* d_randerMap,float* d_randerCount,
    int DEST_WIDTH_, 
    int DEST_HEIGHT_, 
    int channels,int* sy_begin,int* sx_begin,int d_randerMapWidth,int d_randerMapHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 获取 randerMapWidth 和 randerMapHeight 通过解引用指针

    /*if(x== 4 && y== 4)
    {
        printf("x:%d y:%d\n",x,y);
        printf("randerMapWidth:%d randerMapHeight:%d\n",d_randerMapWidth,d_randerMapHeight);
    }*/

    if (x < d_rawImageParameter.m_xLensNum && y < d_rawImageParameter.m_yLensNum)
    {
        // 使用线性索引来访问 RanderMapPatch
        RanderMapPatch patch = d_ppRanderMapPatch[y * d_rawImageParameter.m_xLensNum + x];

        // 计算在输出图像中的起始点
        int sy_b = patch.sy - DEST_HEIGHT_ / 2 - sy_begin[0];
        int sx_b = patch.sx - DEST_WIDTH_ / 2 - sx_begin[0];

        //printf("x:%d y:%d sx:%d sy:%d\n",x,y,sx_b,sy_b);

        // 处理补丁
        for (int py = 0; py < DEST_HEIGHT_; ++py)
        {
            for (int px = 0; px < DEST_WIDTH_; ++px)
            {
                int rander_x = sx_b + px;
                int rander_y = sy_b + py;


                if (rander_x >= 0 && rander_x < d_randerMapWidth && rander_y >= 0 && rander_y < d_randerMapHeight)
                {
                    // 将补丁添加到渲染图
                    for (int c = 0; c < channels; ++c)
                    {
                        atomicAdd(&d_randerMap[(rander_y * d_randerMapWidth + rander_x) * channels + c], patch.simg[(py * DEST_WIDTH_ + px) * channels + c]);
                    }

                    // 统计每个像素被多少个补丁贡献
                    atomicAdd(&d_randerCount[rander_y * d_randerMapWidth + rander_x], 1.0f);
                }
            }
        }
    }
}


__global__ void normalizeKernel(float* d_randerMap,float* d_randerCount,int channels,int d_randerMapWidth,int d_randerMapHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 获取宽度和高度，通过解引用指针
    /*if(x== 100 && y== 100)
    {
        printf("x:%d y:%d\n",x,y);
        printf("randerMapWidth:%d randerMapHeight:%d\n",d_randerMapWidth,d_randerMapHeight);
    }*/

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


    if (x < d_rawImageParameter.m_xLensNum && y < d_rawImageParameter.m_yLensNum &&
        x >= 0 && y >= 0) // Ensure x and y are within valid bounds
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
    int patchWidth, int patchHeight, int Channels,float* d_data )
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

    if(xAdjusted == 21 && yAdjusted == 12 && c == 0 && i_start == 0 && j_start == 0)
    {
        printf("starty:%d startx:%d,d_srcImg:%f,blocksize%d\n",starty,startx,d_srcImg[0]*255,blockSize);
    }


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
    

    if(xAdjusted == 21 && yAdjusted == 12 && c == 0 && i_start == 0 && j_start == 0)
    {
        for(int i =0 ; i < 6; i++)
        {
            for(int j =0 ; j < 6; j++)
            {
                printf("d_simg[%d][%d]:%f\n",i,j,d_simg[(i * patchWidth + j) * Channels + c]*255);
            }
        }
    }

    if (xAdjusted == 21 && yAdjusted == 12 && i_start == 0 && j_start == 0 && c == 0) {
        for (int j = 0; j < blockSize; ++j) {
            for (int i = 0; i < blockSize; ++i) {
                for (int cc = 0; cc < Channels; ++cc) {
                    int srcIdx = (j * blockSize + i) * Channels + cc;
                    int dstIdx = (j * blockSize + i) * Channels + cc;
                    d_data[dstIdx] = d_srcImg[srcIdx];
                }
            }
        }
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
        Channels,
        d_data
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Process Patch Kernel Time: %f ms\n", ms); 
    /*int numPatches = rawImageParameter.m_yLensNum * rawImageParameter.m_xLensNum;
    int patchX = 20;
    int patchY = 20;
    int patchIndex = patchY * rawImageParameter.m_xLensNum + patchX;
    RanderMapPatch* h_patchArray = new RanderMapPatch[numPatches];
    cudaMemcpy(h_patchArray, d_ppRanderMapPatch, sizeof(RanderMapPatch) * numPatches, cudaMemcpyDeviceToHost);
    float* d_patchImg = h_patchArray[patchIndex].simg;
    float* h_patchImg = new float[DEST_WIDTH * DEST_HEIGHT * Channels];
    cudaMemcpy(h_patchImg, d_patchImg, sizeof(float) * DEST_WIDTH * DEST_HEIGHT * Channels, cudaMemcpyDeviceToHost);
    
    // 4. 将 float* 数据转为 cv::Mat（CV_32FC3）
    cv::Mat patchMat(DEST_HEIGHT, DEST_WIDTH, CV_32FC3);
    std::memcpy(patchMat.data, h_patchImg, sizeof(float) * DEST_WIDTH * DEST_HEIGHT * Channels);

    // 5. 可选：转换为 8-bit 保存
    cv::Mat patch8U;
    patchMat.convertTo(patch8U, CV_8UC3, 255.0); // 如果值范围在 0~1

    // 6. 保存图像
    cv::imwrite("gpu_patch_20_20.png", patch8U);
    float* h_data = new float[9* 9* 3];

    // 2. 从 device 拷贝数据到 host
    cudaMemcpy(h_data, d_data, sizeof(float) * 9 * 9 * 3, cudaMemcpyDeviceToHost);
    
    // 3. 将数据写入 OpenCV 的 Mat 中（CV_32FC3）
    cv::Mat image_float(9, 9, CV_32FC3, h_data);
    
    // 4. 转换为 8 位无符号整型用于保存（假设数据在 [0, 1] 范围内）
    cv::Mat image_uchar;
    image_float.convertTo(image_uchar, CV_8UC3, 255.0);
    
    // 5. 写入图像
    cv::imwrite("gpu_patch_9_9.png", image_uchar);*/

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
        /*CUDA_CHECK(cudaMalloc((void**)&h_randerMapWidth, sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_randerMapWidth, &h_randerMapWidth, sizeof(int*)));

        CUDA_CHECK(cudaMalloc((void**)&h_randerMapHeight, sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_randerMapHeight, &h_randerMapHeight, sizeof(int*)));*/
    
    
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

    accumulateKernel<<<gridSize, blockSize>>>(d_ppRanderMapPatch,d_randerMap,d_randerCount,DEST_WIDTH, DEST_HEIGHT, Channels,d_sy_begin,d_sx_begin,randerMapWidthVal_,randerMapHeightVal); // 3通道
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
    normalizeKernel<<<gridSize, blockSize>>>(d_randerMap,d_randerCount,Channels,randerMapWidthVal_,randerMapHeightVal); // 3通道
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