/*!
 * \class 
 *
 * \brief ���ݲ����࣬���ݱ궨��ͼ����������ȷ��һЩ���ڼ����Ӳ�Ĳ���
 *
 * \author liuqian
 * \date ʮһ�� 2017
 */

 #ifndef __DATAPARAMETER_H__
 #define __DATAPARAMETER_H__
 
 #include "CommFunc.h"
 #include "ImageRander.h"
 #include <cuda_runtime.h>
 #include <iomanip>
#include <PvSampleUtils.h>
#include <PvDevice.h>
#include <PvDeviceGEV.h>
#include <PvDeviceU3V.h>
#include <PvStream.h>
#include <PvStreamGEV.h>
#include <PvStreamU3V.h>
#include <PvPipeline.h>
#include <PvBuffer.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <sys/types.h>
 
 #define CUDA_CHECK(call) {                                                   \
	 cudaError_t err = call;                                                   \
	 if (err != cudaSuccess) {                                                 \
		 fprintf(stderr, "CUDA error in function '%s' at %s:%d (call: %s):\n", \
				 __func__, __FILE__, __LINE__, #call);                         \
		 fprintf(stderr, "Error code: %d\n", err);                              \
		 fprintf(stderr, "Error message: %s\n", cudaGetErrorString(err));       \
		 exit(EXIT_FAILURE);                                                   \
	 }                                                                          \
 }
 
 
 
 #define NEIGHBOR_MATCH_LENS_NUM 6 //����ƥ��͸���ĸ���

 struct CudaPoint2f {
    float x, y;

    // 默认构造函数
    __device__ __host__ CudaPoint2f() : x(0), y(0) {}

    // 带参数的构造函数
    __device__ __host__ CudaPoint2f(float x, float y) : x(x), y(y) {}
};
 
 struct MatchNeighborLens
 {
	  MatchNeighborLens()
		 :m_centerPosY(0.0), m_centerPosX(0.0), m_centerDis(0.0), m_centerIndex(0){
	 }
	 MatchNeighborLens(double centerPosY, double centerPosX, double centerDis, int centerIndex)
	 {
		 m_centerPosY = centerPosY; 
		 m_centerPosX = centerPosX; 
		 m_centerDis = centerDis; 
		 m_centerIndex = centerIndex;
	 }
	 MatchNeighborLens(const MatchNeighborLens &srcNeighborLens)
	 {
		 m_centerPosY = srcNeighborLens.m_centerPosY;
		 m_centerPosX = srcNeighborLens.m_centerPosX;
		 m_centerDis = srcNeighborLens.m_centerDis;
		 m_centerIndex = srcNeighborLens.m_centerIndex;
	 }
	 float m_centerPosY; //͸������λ��y����
	 float m_centerPosX; //͸������x����
	 float m_centerDis; //͸����ľ���
	 int m_centerIndex; //͸����ǩ�����ڵڼ���͸����
 };
 
 
 struct MicroImageParameter
 {
	 float m_circleDiameter; //΢͸��ͼ��СԲֱ��
	 float m_circleNarrow; //ʵ�ʳ�����Ϊ�˷�ֹСԲ�߽��ϵ�����ʵ�ʴ�����СԲ���������С
	 float m_radiusDisEqu; //СԲ�뾶�����ƽ��
	 cv::Point2d **m_ppLensCenterPoints; //ÿ��΢͸��ͼ��������Rawͼ���е�λ��
	 int **m_ppPixelsMappingSet; //����֮���ӳ�伯�ϣ�����Rawͼ����ÿ�����ر���ϱ�ǩ���ڵڼ���΢͸��ͼ��
	 MatchNeighborLens ***m_ppMatchNeighborLens; //�洢ÿһ��͸����Χ����ƥ��͸������Ϣ
 };
 
 
 struct RawImageParameter
 {
	 int m_yLensNum; //y�����ϵ���Ч΢͸��ͼ�����
	 int m_xLensNum; //x�����ϵ���Ч΢͸��ͼ�����
	 int m_yCenterBeginOffset; //Y�����ϴ�������ƫ�ƶ�����͸��
	 int m_xCenterBeginOffset; //X�����ϴ�������ƫ�ƶ�����͸��
	 int m_yCenterEndOffset; //Y�����ϴ�������ƫ�ƶ�����͸��
	 int m_xCenterEndOffset; //X�����ϴ�������ƫ�ƶ�����͸��
	 int m_recImgHeight; //��ͼ�����ͼ��߶�
	 int m_recImgWidth; //��ͼ�����ͼ�����
	 int m_srcImgHeight; //����ͼ���ͼ��߶�
	 int m_srcImgWidth; //����ͼ���ͼ�����
	 int m_yPixelBeginOffset; //y�����ϵ�����ƫ��ֵ
	 int m_xPixelBeginOffset; //x�����ϵ�����ƫ��ֵ
 };
 
 
 struct DisparityParameter
 {
	 int m_dispMin; //�Ӳ���Сֵ
	 int m_dispMax; //�Ӳ����ֵ
	 float m_dispStep; //�Ӳ��������
	 int m_disNum; //�ܹ����Ӳ�label��Ŀ
 };
 
 struct FilterPatameter
 {
	 cv::Mat *m_pValidNeighborPixelsNum; //ÿ���������˲�����������Ч���ص���Ŀ
	 cv::Mat *m_pValidPixelsMask; //�����Щ��������Ч���أ�΢͸�������ڵ����أ�
	 cv::Mat m_filterKnernel; //�Զ����˲���
 };
 
 struct FilterParameterDevice {
	 float* d_validNeighborPixelsNum;
	 float* d_validPixelsMask;
	 float* d_filterKernel;
 };
 
 struct MicroImageParameterDevice {
	 float m_circleDiameter;   // 圆形直径
	 float m_circleNarrow;     // 圆形狭窄
	 float m_radiusDisEqu;     // 圆形半径差异
 
	 CudaPoint2f* m_ppLensCenterPoints;  // 存储在 GPU 上的 Lens Center Points
	 int* m_ppPixelsMappingSet;           // 存储在 GPU 上的 Pixels Mapping Set
	 MatchNeighborLens* m_ppMatchNeighborLens;  // 存储在 GPU 上的 Match Neighbor Lens
 };
 
 
 class DataParameter
 {
 public:
	 //���ݲ�����ʼ������
	 DataParameter();
	 DataParameter(std::string dataFolderName, std::string centerPointFileName, std::string inputImgName,
		 int yCenterBeginOffset = 2, int xCenterBeginOffset = 2, int yCenterEndOffset = 2, int xCenterEndOffset = 2,
		 int filterRadius = 4, float circleDiameter = 34.0, float circleNarrow = 1.5, int dispMin = 5, int dispMax = 13, float dispStep = 0.5);
	 ~DataParameter();
	 void init(std::string centerPointFileName, std::string dataFolderPath, std::string inputImgName,
		 int yCenterBeginOffset = 2, int xCenterBeginOffset = 2, int yCenterEndOffset = 2, int xCenterEndOffset = 2,
		 int filterRadius = 4, float circleDiameter = 34.0, float circleNarrow = 1.5, int dispMin = 5, int dispMax = 13, float dispStep = 0.5);
	 
	 //���ݲ������ò���
	 void dispSet(int dispMin = 5, int dispMax = 13, float dispStep = 0.5);
	 void srcImageSet(std::string dataFolderPath, std::string inputImgName);
	 void mapToGPU();
	 void UpdateImgToGPU();
	 void UploadtoGPU() const;
 
	 RawImageParameter getRawImageParameter() const
	 {
		 return m_rawImageParameter;
	 };
	 MicroImageParameter getMicroImageParameter() const
	 {
		 return m_microImageParameter;
	 };
	 DisparityParameter getDisparityParameter() const
	 {
		 return m_disparityParameter;
	 };
	 FilterPatameter getFilterPatameter() const
	 {
		 return m_filterPatameter;
	 };
 
	 //ͼ����Ϣ
	 cv::Mat m_inputImg; //�����Rawͼ��
	 cv::Mat m_inputImgRec; //�������Rawͼ���ͼ����
	 std::string m_folderPath; //�������ݴ���ļ�����
	 //������Ҫ�����õ����ŵ����в��ּ���Ч��
 private:
	 void lensCenterPointsInit(std::string dataFolderPath, std::string centerPointFileName); //͸�����ĵ��ʼ��
	 void validLensCenterInit(int yCenterBeginOffset, int xCenterBeginOffset, int yCenterEndOffset, int xCenterEndOffset); //�����Щ͸��������Ч
	 void imageBaseMessageInit(std::string inputImgName, int filterRadius, float circleDiameter,
		 float circleNarrow, int dispMin, int dispMax, float dispStep); //ͼ�������Ϣ���Ӳ���Ϣ��ʼ��
	 void generatePixelsMappingSet(); //�������ص�ӳ�伯��
	 void generatePixelsMappingSet(int y, int x); //�������ص�ӳ�伯��--����ÿ����ͼ���cost
	 void generateNeighborCenterPoints(); //����ÿ�����ĵ���Χ�Ķ�Ӧ���ĵ�
	 void generateNeighborCenterPoints(int y, int x);//����ÿ�����ĵ���Χ�Ķ�Ӧ���ĵ�
	 void generateValidPoints(); //����ÿ��СԲ�����أ�����Χ��Ч�����
	 void generateValidPoints(int y, int x); //����ÿ��СԲ�����أ�����Χ��Ч���
	 void validPointsBoundaryRepair(); //����Ч��ı߽��������ؽ�����1��������ֹ���ֳ�0����
 
	 
	 RawImageParameter m_rawImageParameter; //Rawͼ�������Ϣ
	 MicroImageParameter m_microImageParameter; //΢͸��ͼ�������Ϣ
	 DisparityParameter m_disparityParameter; //�Ӳ���Ϣ
	 FilterPatameter m_filterPatameter; //�˲�������Ϣ
 
	 
	 int m_lensArrageMode; //�ж�΢͸��ͼ����Ų����� == 1��ʾ��0��X����ͻ����0��ʾ��0��X���갼��
	 int m_filterRadius; //���Ӳ���˲��뾶����Ҫ�Ƿ����˲��ۺϣ�
 };
 
 extern __constant__ RawImageParameter d_rawImageParameter;
 extern __constant__ DisparityParameter d_disparityParameter;
 extern FilterParameterDevice* d_filterPatameterDevice; 
 extern  MicroImageParameterDevice* d_microImageParameter; 
 extern __device__ float* d_costVol;
 extern __device__ float* d_costVolFiltered;
 extern  float* d_rawDisp;
 extern  float* d_rawDisp_temp;
 extern __device__ float* d_ppLensMeanDisp;
 extern __device__ float* d_renderCache;
 extern  float* d_inputImg;
 extern  float* d_inputImgRec;
 extern  float* d_grayImg;
 extern  float* d_gradImg;
 extern RanderMapPatch* d_ppRanderMapPatch;
 extern __device__ float* d_tmp;
 extern __device__ float* d_simg;
 extern int* d_sx_begin;
 extern int* d_sy_begin;
 extern int* d_sx_end;
 extern int* d_sy_end;
 extern __device__ int *d_randerMapWidth, *d_randerMapHeight;
 extern __constant__ float d_filterRadius;
 extern __constant__ float d_fltMax;
 extern __constant__ int d_meanDispLenRadius;
 extern __constant__ int d_patchScale9;
 extern __constant__ float d_randerScale;
 extern __constant__ int d_destWidth;
 extern __constant__ int d_destHeight;
 extern float* d_data; 




 #define BUFFER_COUNT (16)
extern PvString lConnectionID;
extern PvResult lResult;
extern PvDevice *lDevice;
extern PvStream *lStream;
extern PvDeviceGEV *lDeviceGEV;
extern PvPipeline *lPipeline;
extern PvGenParameterArray *lDeviceParams;
extern PvGenCommand *lStart;
extern PvGenCommand *lStop;
extern PvBuffer *lBuffer;
extern PvResult lOperationResult;

 #endif