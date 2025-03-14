/*!
 * \class ImageRander
 *
 * \brief 根据Raw视差对子孔径图像的渲染
 *
 * \author liuqian
 * \date 一月 2018
 */


#ifndef __IMAGERANDER_H_
#define __IMAGERANDER_H_

#include "CommFunc.h"
#include "DataDeal.h"

class DataParameter;
struct RawImageParameter;
struct MicroImageParameter;

struct RanderMapPatch
{
	int sy, sx; //小的渲染图在全局渲染图中的起始位置
	cv::Mat simg;
};

class ImageRander : public DataDeal
{
public:
	ImageRander();
	~ImageRander();

	void imageRanderWithMask(const DataParameter &dataParameter, cv::Mat &rawDisp, cv::Mat *confidentMask);//对带置信度mask的情况进行子孔径渲染
	void imageRanderWithOutMask(const DataParameter &dataParameter, cv::Mat &rawDisp);//对没有置信度mask的情况进行子孔径渲染
private:
	void imageRander(float **ppLensMeanDisp, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, cv::Mat &randerImg, cv::Mat &destImg);
	void imageRanderRepair(const RawImageParameter &rawImageParameter, cv::Mat &randerMap, cv::Mat &repairMap, RanderMapPatch **ppRanderMapPatch, int sx_begin, int sy_begin);//去除边界黑色空洞
	void outputSparseSceneDepth(string folderName, cv::Mat &sceneSparseDepth, cv::Mat &sceneDepthMask);
};

#endif