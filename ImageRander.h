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
 #include <opencv2/opencv.hpp>
 #include <PvSampleUtils.h>
 #include <PvSystem.h>
 #include <PvDevice.h>
 #include <PvDeviceGEV.h>
 #include <PvDeviceU3V.h>
 #include <PvStream.h>
 #include <PvStreamGEV.h>
 #include <PvStreamU3V.h>
 #include <PvPipeline.h>
 #include <PvBuffer.h>
 //修改了WIDTH和HEIGHT,此处参数只能影响到渲染图的生成效果
 #define MEAN_DISP_LEN_RADIUS 18//平均距离长度 8 注意该参数在需要让计算深度的点尽量都在一个圆内(方型)  10
 #define PATCH_SCALE9 9//路径比例 9
 #define RANDER_SCALE 0.9//渲染比例 render  0.35
 #define DEST_WIDTH 44//38 27 44
 #define DEST_HEIGHT 44//38 27 44
 
 
 class DataParameter;
 struct RawImageParameter;
 struct MicroImageParameter;
 
 struct RanderMapPatch
 {
	 int sy, sx;  // 记录图像的起始位置
	 float* simg; // 存储图像数据的指针，采用原始数组代替 cv::Mat
 };
 
 void saveSingleChannelGpuMemoryAsImage(float* d_data, int width, int height,const std::string& filename);
 void saveThreeChannelGpuMemoryAsImage(float* d_data, int width, int height,const std::string& filename);
 class ImageRander : public DataDeal
 {
 public:
	 ImageRander();
	 ~ImageRander();
 
	 void imageRanderWithMask(const DataParameter &dataParameter, cv::Mat &rawDisp, cv::Mat *confidentMask);//�Դ����Ŷ�mask����������ӿ׾���Ⱦ
	 void imageRanderWithOutMask(const DataParameter &dataParameter);//��û�����Ŷ�mask����������ӿ׾���Ⱦ
 private:
	 void imageRander(const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, float *d_randerImg);
	 
void imageRander(const RawImageParameter &rawImageParameter, 
    const MicroImageParameter &microImageParameter,float* d_input,int Channels);

 //void imageRander(float **ppLensMeanDisp, const RawImageParameter &rawImageParameter, const MicroImageParameter &microImageParameter, cv::Mat &randerImg, cv::Mat &destImg);
	 void imageRanderRepair(const RawImageParameter &rawImageParameter, cv::Mat &randerMap, cv::Mat &repairMap, RanderMapPatch **ppRanderMapPatch, int sx_begin, int sy_begin);//ȥ���߽��ɫ�ն�
	 void outputSparseSceneDepth(string folderName, cv::Mat &sceneSparseDepth, cv::Mat &sceneDepthMask);
 };
 
 #endif