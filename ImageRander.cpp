#include "ImageRander.h"
#include "DataParameter.h"
#include <iomanip>
#include <opencv2/cudaimgproc.hpp> // CUDA 图像处理模块
#include <opencv2/cudaarithm.hpp>   // CUDA 算术操作模块

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




void ImageRander::imageRanderWithOutMask(const DataParameter &dataParameter, cv::Mat &rawDisp) {
    RawImageParameter rawImageParameter = dataParameter.getRawImageParameter();
    MicroImageParameter microImageParameter = dataParameter.getMicroImageParameter();
    DisparityParameter disparityParameter = dataParameter.getDisparityParameter();

    // 将 rawDisp 转换为浮点类型
    cv::Mat tmpRawDisp;
    rawDisp.convertTo(tmpRawDisp, CV_32F, disparityParameter.m_dispStep, disparityParameter.m_dispMin);

    // 分配内存用于存储每个透镜的平均视差
    float **ppLensMeanDisp = new float*[rawImageParameter.m_yLensNum];
    for (int i = 0; i < rawImageParameter.m_yLensNum; i++) {
        ppLensMeanDisp[i] = new float[rawImageParameter.m_xLensNum];
    }

    // 使用 CUDA 加速计算每个透镜的平均视差
    #pragma omp parallel for
    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++) {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++) {
            Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];
            int x_begin = curCenterPos.x - rawImageParameter.m_xPixelBeginOffset - MEAN_DISP_LEN_RADIUS;
            int y_begin = curCenterPos.y - rawImageParameter.m_yPixelBeginOffset - MEAN_DISP_LEN_RADIUS;

            // 检查边界，确保矩形区域在图像范围内
            x_begin = max(0, x_begin);
            y_begin = max(0, y_begin);
            int width = min(MEAN_DISP_LEN_RADIUS * 2 + 1, tmpRawDisp.cols - x_begin);
            int height = min(MEAN_DISP_LEN_RADIUS * 2 + 1, tmpRawDisp.rows - y_begin);

            if (width > 0 && height > 0) {
                // 提取当前区域的视差图
                cv::Mat srcCost = tmpRawDisp(cv::Rect(x_begin, y_begin, width, height));

                // 使用 CUDA 加速计算均值
                cv::cuda::GpuMat gpuSrcCost;
                gpuSrcCost.upload(srcCost); // 上传数据到 GPU

                // 计算总和
                cv::Scalar sumValue = cv::cuda::sum(gpuSrcCost);

                // 计算均值
                double meanValue = sumValue[0] / (srcCost.rows * srcCost.cols);
                ppLensMeanDisp[y][x] = std::max(meanValue, (double)(disparityParameter.m_dispMin));
            } else {
                ppLensMeanDisp[y][x] = disparityParameter.m_dispMin; // 如果区域无效，使用最小视差
            }
        }
    }

    // 渲染开始
    cv::Mat randerMapinput, randerSceneMap, finalRanderMap;
    dataParameter.m_inputImgRec.convertTo(randerMapinput, CV_64FC3);

    // 调用 imageRander 函数进行渲染
    imageRander(ppLensMeanDisp, rawImageParameter, microImageParameter, randerMapinput, randerSceneMap);

    // 保存渲染结果
    std::string storeName = dataParameter.m_folderPath + "/randerSceneMap.bmp";
    randerSceneMap.convertTo(finalRanderMap, CV_8UC3);
    imwrite(storeName, finalRanderMap);

    // 渲染视差图
    cv::Mat sceneDisp = cv::Mat::zeros(rawDisp.rows, rawDisp.cols, CV_64FC1);
    rawDisp.convertTo(sceneDisp, CV_64FC1, disparityParameter.m_dispStep, disparityParameter.m_dispMin);

    cv::Mat randerDispMap, randerSparseDispMap;
    imageRander(ppLensMeanDisp, rawImageParameter, microImageParameter, sceneDisp, randerDispMap);
    randerDispMap.copyTo(randerSparseDispMap);

    // 保存渲染后的视差图
    storeName = dataParameter.m_folderPath + "/randerDispMap.bmp";
    dispMapShowForColor(storeName, randerDispMap);

    // 释放内存
    for (int i = 0; i < rawImageParameter.m_yLensNum; i++) {
        delete[] ppLensMeanDisp[i];
    }
    delete[] ppLensMeanDisp;
}


void ImageRander::imageRander(float **ppLensMeanDisp, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, cv::Mat &randerImg, cv::Mat &destImg) {
    // 使用 std::vector 管理内存，避免手动分配和释放
    vector<vector<RanderMapPatch>> ppRanderMapPatch(
        rawImageParameter.m_yLensNum,
        vector<RanderMapPatch>(rawImageParameter.m_xLensNum)
    );

    // 并行化外层和内层循环
    #pragma omp parallel for collapse(2)
    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++) {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++) {
            int blockSize = fabs(std::round(ppLensMeanDisp[y][x]));
            Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];
            int starty = curCenterPos.y - blockSize / 2 - rawImageParameter.m_yPixelBeginOffset;
            int startx = curCenterPos.x - blockSize / 2 - rawImageParameter.m_xPixelBeginOffset;

            // 检查边界，确保矩形区域在图像范围内
            startx = max(0, startx);
            starty = max(0, starty);
            int width = min(blockSize, randerImg.cols - startx);
            int height = min(blockSize, randerImg.rows - starty);

            if (width > 0 && height > 0) {
                cv::Mat srcImg = randerImg(cv::Rect(startx, starty, width, height));
                ppRanderMapPatch[y][x].sy = curCenterPos.y;
                ppRanderMapPatch[y][x].sx = curCenterPos.x;

                // 使用 CPU 进行图像缩放
                cv::Mat tmp;
                cv::resize(srcImg, tmp, cv::Size(DEST_WIDTH, DEST_HEIGHT), 0, 0, cv::INTER_LINEAR);
                cv::flip(tmp, ppRanderMapPatch[y][x].simg, -1);
            }
        }
    }

    // 计算渲染图的范围
    int sx_begin = INT_MAX, sy_begin = INT_MAX;
    int sx_end = INT_MIN, sy_end = INT_MIN;

    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++) {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++) {
            sy_begin = min(sy_begin, ppRanderMapPatch[y][x].sy - ppRanderMapPatch[y][x].simg.rows / 2);
            sx_begin = min(sx_begin, ppRanderMapPatch[y][x].sx - ppRanderMapPatch[y][x].simg.cols / 2);
            sy_end = max(sy_end, ppRanderMapPatch[y][x].sy + ppRanderMapPatch[y][x].simg.rows / 2);
            sx_end = max(sx_end, ppRanderMapPatch[y][x].sx + ppRanderMapPatch[y][x].simg.cols / 2);
        }
    }

    int randerMapWidth = sx_end - sx_begin + 1;
    int randerMapHeight = sy_end - sy_begin + 1;

    // 创建渲染图和计数图
    cv::Mat randerMap = cv::Mat::zeros(randerMapHeight, randerMapWidth, randerImg.type());
    cv::Mat randerCount = cv::Mat::zeros(randerMapHeight, randerMapWidth, CV_64FC1);
    cv::Mat tmpCount = cv::Mat::ones(DEST_HEIGHT, DEST_WIDTH, CV_64FC1); // 确保大小匹配

    // 使用 CUDA 加速矩阵操作
    cv::cuda::GpuMat gpuRanderMap, gpuRanderCount;
    gpuRanderMap.upload(randerMap);
    gpuRanderCount.upload(randerCount);

    for (int y = rawImageParameter.m_yCenterBeginOffset; y < rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset; y++) {
        for (int x = rawImageParameter.m_xCenterBeginOffset; x < rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset; x++) {
            int sy_b = ppRanderMapPatch[y][x].sy - ppRanderMapPatch[y][x].simg.rows / 2 - sy_begin;
            int sx_b = ppRanderMapPatch[y][x].sx - ppRanderMapPatch[y][x].simg.cols / 2 - sx_begin;

            // 提取 ROI，确保大小与 tmpCount 和 gpuPatch 匹配
            cv::Rect roi(sx_b, sy_b, DEST_WIDTH, DEST_HEIGHT);
            cv::cuda::GpuMat gpuRanderMapROI = gpuRanderMap(roi);
            cv::cuda::GpuMat gpuRanderCountROI = gpuRanderCount(roi);

            // 上传 gpuPatch 和 tmpCount
            cv::cuda::GpuMat gpuPatch;
            gpuPatch.upload(ppRanderMapPatch[y][x].simg);

            cv::cuda::GpuMat gpuTmpCount;
            gpuTmpCount.upload(tmpCount);

            // 执行 CUDA 加法
            cv::cuda::add(gpuRanderMapROI, gpuPatch, gpuRanderMapROI);
            cv::cuda::add(gpuRanderCountROI, gpuTmpCount, gpuRanderCountROI);
        }
    }

    // 下载结果到 CPU
    gpuRanderMap.download(randerMap);
    gpuRanderCount.download(randerCount);

    // 归一化渲染图
    for (int y = 0; y < randerMapHeight; y++) {
        double *yRanderData = (double *)randerMap.ptr<double>(y);
        double *yRanderCount = (double *)randerCount.ptr<double>(y);
        for (int x = 0; x < randerMapWidth; x++) {
            if (yRanderCount[x] >= 1.0) {
                if (randerMap.channels() == 3) {
                    yRanderData[3 * x] /= yRanderCount[x];
                    yRanderData[3 * x + 1] /= yRanderCount[x];
                    yRanderData[3 * x + 2] /= yRanderCount[x];
                } else {
                    yRanderData[x] /= yRanderCount[x];
                }
            }
        }
    }

    // 将 std::vector<std::vector<RanderMapPatch>> 转换为 RanderMapPatch**
    RanderMapPatch **ppRanderMapPatchArray = new RanderMapPatch*[rawImageParameter.m_yLensNum];
    for (int i = 0; i < rawImageParameter.m_yLensNum; i++) {
        ppRanderMapPatchArray[i] = ppRanderMapPatch[i].data();
    }

    // 修复渲染图并调整大小
    cv::Mat repairMap;
    imageRanderRepair(rawImageParameter, randerMap, repairMap, ppRanderMapPatchArray, sx_begin, sy_begin);
    cv::resize(repairMap, destImg, cv::Size(0, 0), RANDER_SCALE, RANDER_SCALE, INTER_CUBIC);

    // 释放内存
    delete[] ppRanderMapPatchArray;
}

/*
void ImageRander::imageRander(float **ppLensMeanDisp, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, cv::Mat &randerImg, cv::Mat &destImg)
{
	const int BLOCK_SIZE = 10; 


	// 获取微透镜中心点的数量
	const int yLensNum = rawImageParameter.m_yLensNum;
	const int xLensNum = rawImageParameter.m_xLensNum;

	// 计算最终拼接图像的大小
	int destImgWidth = xLensNum * DEST_WIDTH;
	int destImgHeight = yLensNum * DEST_HEIGHT;
	destImg = cv::Mat::zeros(destImgHeight, destImgWidth, randerImg.type());

	// 遍历每个微透镜中心点
	for (int y = 0; y < yLensNum; y++)
	{
		for (int x = 0; x < xLensNum; x++)
		{
			// 获取当前微透镜的中心点
			cv::Point2d &curCenterPos = microImageParameter.m_ppLensCenterPoints[y][x];

			// 计算块的起始位置
			int starty = static_cast<int>(curCenterPos.y - BLOCK_SIZE / 2);
			int startx = static_cast<int>(curCenterPos.x - BLOCK_SIZE / 2);

			// 检查边界，确保矩形区域在图像范围内
			startx = std::max(0, startx);
			starty = std::max(0, starty);
			int endx = std::min(randerImg.cols, startx + BLOCK_SIZE);
			int endy = std::min(randerImg.rows, starty + BLOCK_SIZE);

			// 计算实际的矩形区域大小
			int width = endx - startx;
			int height = endy - starty;

			// 如果矩形区域有效，则提取块
			if (width > 0 && height > 0)
			{
				cv::Mat srcBlock = randerImg(cv::Rect(startx, starty, width, height));

				// 如果需要调整块的大小，可以使用 cv::resize
				if (width != DEST_WIDTH || height != DEST_HEIGHT)
				{
					cv::resize(srcBlock, srcBlock, cv::Size(DEST_WIDTH, DEST_HEIGHT));
				}

				// 将块拼接到目标图像中
				int destY = y * DEST_HEIGHT;
				int destX = x * DEST_WIDTH;
				srcBlock.copyTo(destImg(cv::Rect(destX, destY, DEST_WIDTH, DEST_HEIGHT)));
			}
		}
	}
}*/
void ImageRander::imageRanderRepair(const RawImageParameter &rawImageParameter, cv::Mat &randerMap, cv::Mat &repairMap, RanderMapPatch **ppRanderMapPatch, int sx_begin, int sy_begin) {
    int randerMapHeight = randerMap.rows;
    int randerMapWidth = randerMap.cols;

    int left = rawImageParameter.m_xCenterBeginOffset;
    int right = rawImageParameter.m_xLensNum - rawImageParameter.m_xCenterEndOffset - 1;
    int top = rawImageParameter.m_yCenterBeginOffset;
    int below = rawImageParameter.m_yLensNum - rawImageParameter.m_yCenterEndOffset - 1;

    int x_left = 0;
    int x_right = randerMapWidth;
    int y_top = 0;
    int y_below = randerMapHeight;

    // 预计算 left 和 right 的 sx 和 simg.cols
    vector<int> left_sx(below - top + 1);
    vector<int> left_cols(below - top + 1);
    vector<int> right_sx(below - top + 1);
    vector<int> right_cols(below - top + 1);

    #pragma omp parallel for
    for (int y = top; y <= below; y++) {
        left_sx[y - top] = ppRanderMapPatch[y][left].sx;
        left_cols[y - top] = ppRanderMapPatch[y][left].simg.cols;
        right_sx[y - top] = ppRanderMapPatch[y][right].sx;
        right_cols[y - top] = ppRanderMapPatch[y][right].simg.cols;
    }

    // 计算 x_left 和 x_right
    #pragma omp parallel for reduction(max:x_left) reduction(min:x_right)
    for (int y = top; y <= below; y++) {
        x_left = max(x_left, int(left_sx[y - top] - left_cols[y - top] / 2 - sx_begin));
        x_right = min(x_right, int(right_sx[y - top] - right_cols[y - top] / 2 - sx_begin + right_cols[y - top]));
    }

    // 预计算 top 和 below 的 sy 和 simg.rows
    vector<int> top_sy(right - left + 1);
    vector<int> top_rows(right - left + 1);
    vector<int> below_sy(right - left + 1);
    vector<int> below_rows(right - left + 1);

    #pragma omp parallel for
    for (int x = left; x <= right; x++) {
        top_sy[x - left] = ppRanderMapPatch[top][x].sy;
        top_rows[x - left] = ppRanderMapPatch[top][x].simg.rows;
        below_sy[x - left] = ppRanderMapPatch[below][x].sy;
        below_rows[x - left] = ppRanderMapPatch[below][x].simg.rows;
    }

    // 计算 y_top 和 y_below
    #pragma omp parallel for reduction(max:y_top) reduction(min:y_below)
    for (int x = left; x <= right; x++) {
        y_top = max(y_top, int(top_sy[x - left] - top_rows[x - left] / 2 - sy_begin));
        y_below = min(y_below, int(below_sy[x - left] - below_rows[x - left] / 2 - sy_begin + below_rows[x - left]));
    }

    // 检查边界，确保提取的区域在图像范围内
    x_left = max(0, x_left);
    y_top = max(0, y_top);
    x_right = min(randerMapWidth, x_right);
    y_below = min(randerMapHeight, y_below);

    // 使用 CUDA 加速矩阵拷贝
    cv::cuda::GpuMat gpuRanderMap, gpuRepairMap;
    gpuRanderMap.upload(randerMap);

    if (x_right > x_left && y_below > y_top) {
        cv::Rect roi(x_left, y_top, x_right - x_left, y_below - y_top);
        gpuRepairMap = gpuRanderMap(roi).clone();
    } else {
        gpuRepairMap = cv::cuda::GpuMat(randerMap.size(), randerMap.type(), cv::Scalar(0));
    }

    // 下载结果到 CPU
    gpuRepairMap.download(repairMap);
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