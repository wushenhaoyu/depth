/*!
 * \class ImageRander
 *
 * \brief ����Raw�Ӳ���ӿ׾�ͼ�����Ⱦ
 *
 * \author liuqian
 * \date һ�� 2018
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
	int sy, sx; //С����Ⱦͼ��ȫ����Ⱦͼ�е���ʼλ��
	cv::Mat simg;
};

class ImageRander : public DataDeal
{
public:
	ImageRander();
	~ImageRander();

	void imageRanderWithMask(const DataParameter &dataParameter, cv::Mat &rawDisp, cv::Mat *confidentMask);//�Դ����Ŷ�mask����������ӿ׾���Ⱦ
	void imageRanderWithOutMask(const DataParameter &dataParameter, cv::Mat &rawDisp);//��û�����Ŷ�mask����������ӿ׾���Ⱦ
private:
	void imageRander(float **ppLensMeanDisp, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, cv::Mat &randerImg, cv::Mat &destImg);
	void imageRanderRepair(const RawImageParameter &rawImageParameter, cv::Mat &randerMap, cv::Mat &repairMap, RanderMapPatch **ppRanderMapPatch, int sx_begin, int sy_begin);//ȥ���߽��ɫ�ն�
	void outputSparseSceneDepth(string folderName, cv::Mat &sceneSparseDepth, cv::Mat &sceneDepthMask);
};

#endif