/*!
 * \class 
 *
 * \brief 数据参数类，根据标定的图像中心数据确定一些便于计算视差的参数
 *
 * \author liuqian
 * \date 十一月 2017
 */

#ifndef __DATAPARAMETER_H__
#define __DATAPARAMETER_H__

#include "CommFunc.h"

#define NEIGHBOR_MATCH_LENS_NUM 6 //相邻匹配透镜的个数

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
	float m_centerPosY; //透镜中心位置y坐标
	float m_centerPosX; //透镜中心x坐标
	float m_centerDis; //透镜间的距离
	int m_centerIndex; //透镜标签（属于第几号透镜）
};

struct RawImageParameter
{
	int m_yLensNum; //y方向上的有效微透镜图像个数
	int m_xLensNum; //x方向上的有效微透镜图像个数
	int m_yCenterBeginOffset; //Y方向上从上至下偏移多少行透镜
	int m_xCenterBeginOffset; //X方向上从左至右偏移多少行透镜
	int m_yCenterEndOffset; //Y方向上从下至上偏移多少行透镜
	int m_xCenterEndOffset; //X方向上从右至左偏移多少行透镜
	int m_recImgHeight; //截图区域的图像高度
	int m_recImgWidth; //截图区域的图像宽度
	int m_srcImgHeight; //输入图像的图像高度
	int m_srcImgWidth; //输入图像的图像宽度
	int m_yPixelBeginOffset; //y方向上的像素偏移值
	int m_xPixelBeginOffset; //x方向上的像素偏移值
};

struct MicroImageParameter
{
	float m_circleDiameter; //微透镜图像小圆直径
	float m_circleNarrow; //实际程序处理为了防止小圆边界上的误差，将实际处理的小圆区域进行缩小
	float m_radiusDisEqu; //小圆半径距离的平方
	cv::Point2d **m_ppLensCenterPoints; //每个微透镜图像中心在Raw图像中的位置
	int **m_ppPixelsMappingSet; //像素之间的映射集合，即给Raw图像上每个像素标记上标签属于第几个微透镜图像
	MatchNeighborLens ***m_ppMatchNeighborLens; //存储每一个透镜周围邻域匹配透镜的信息
};

struct DisparityParameter
{
	int m_dispMin; //视差最小值
	int m_dispMax; //视差最大值
	float m_dispStep; //视差迭代步长
	int m_disNum; //总共的视差label数目
};

struct FilterPatameter
{
	cv::Mat *m_pValidNeighborPixelsNum; //每个像素在滤波核内相邻有效像素的数目
	cv::Mat *m_pValidPixelsMask; //标记哪些像素是有效像素（微透镜区域内的像素）
	cv::Mat m_filterKnernel; //自定义滤波核
};

class DataParameter
{
public:
	//数据参数初始化部分
	DataParameter();
	DataParameter(std::string dataFolderName, std::string centerPointFileName, std::string inputImgName,
		int yCenterBeginOffset = 2, int xCenterBeginOffset = 2, int yCenterEndOffset = 2, int xCenterEndOffset = 2,
		int filterRadius = 4, float circleDiameter = 34.0, float circleNarrow = 1.5, int dispMin = 5, int dispMax = 13, float dispStep = 0.5);
	~DataParameter();
	void init(std::string centerPointFileName, std::string dataFolderPath, std::string inputImgName,
		int yCenterBeginOffset = 2, int xCenterBeginOffset = 2, int yCenterEndOffset = 2, int xCenterEndOffset = 2,
		int filterRadius = 4, float circleDiameter = 34.0, float circleNarrow = 1.5, int dispMin = 5, int dispMax = 13, float dispStep = 0.5);
	
	//数据参数重置部分
	void dispSet(int dispMin = 5, int dispMax = 13, float dispStep = 0.5);
	void srcImageSet(std::string dataFolderPath, std::string inputImgName);

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

	//图像信息
	cv::Mat m_inputImg; //输入的Raw图像
	cv::Mat m_inputImgRec; //对输入的Raw图像截图后结果
	std::string m_folderPath; //定义数据存放文件夹名
	//由于需要经常用到，放到公有部分加速效率
private:
	void lensCenterPointsInit(std::string dataFolderPath, std::string centerPointFileName); //透镜中心点初始化
	void validLensCenterInit(int yCenterBeginOffset, int xCenterBeginOffset, int yCenterEndOffset, int xCenterEndOffset); //标记哪些透镜中心有效
	void imageBaseMessageInit(std::string inputImgName, int filterRadius, float circleDiameter,
		float circleNarrow, int dispMin, int dispMax, float dispStep); //图像基本信息及视差信息初始化
	void generatePixelsMappingSet(); //构造像素的映射集合
	void generatePixelsMappingSet(int y, int x); //构造像素的映射集合--计算每个子图像的cost
	void generateNeighborCenterPoints(); //生成每个中心点周围的对应中心点
	void generateNeighborCenterPoints(int y, int x);//生成每个中心点周围的对应中心点
	void generateValidPoints(); //生成每个小圆中像素，其周围有效点个数
	void generateValidPoints(int y, int x); //生成每个小圆中像素，其周围有效点个
	void validPointsBoundaryRepair(); //对有效点的边界区域像素进行置1处理，防止出现除0现象

	/*
	//如输入参数相关变量
	int m_yCenterBeginOffset; //Y方向上从上至下偏移多少行透镜
	int m_xCenterBeginOffset; //X方向上从左至右偏移多少行透镜
	int m_yCenterEndOffset; //Y方向上从下至上偏移多少行透镜
	int m_xCenterEndOffset; //X方向上从右至左偏移多少行透镜
	int m_yLensNum; //y方向上的有效微透镜图像个数
	int m_xLensNum; //x方向上的有效微透镜图像个数
	int m_recImgHeight; //截图区域的图像高度
	int m_recImgWidth; //截图区域的图像宽度
	int m_srcImgHeight; //输入图像的图像高度
	int m_srcImgWidth; //输入图像的图像宽度
	int m_yPixelBeginOffset; //y方向上的像素偏移值
	int m_xPixelBeginOffset; //x方向上的像素偏移值

	float m_circleDiameter; //微透镜图像小圆直径
	float m_circleNarrow; //实际程序处理为了防止小圆边界上的误差，将实际处理的小圆区域进行缩小
	float m_radiusDisEqu; //小圆半径距离的平方
	*/
	
	RawImageParameter m_rawImageParameter; //Raw图像参数信息
	MicroImageParameter m_microImageParameter; //微透镜图像参数信息
	DisparityParameter m_disparityParameter; //视差信息
	FilterPatameter m_filterPatameter; //滤波参数信息

	/*
	int m_dispMin; //视差最小值
	int m_dispMax; //视差最大值
	float m_dispStep; //视差迭代步长
	//根据中心位置做一些预设值的变量
	int m_disNum; //总共的视差label数目
	*/

	int m_lensArrageMode; //判断微透镜图像的排布类型 == 1表示第0行X坐标突出，0表示第0行X坐标凹陷
	int m_filterRadius; //对视差的滤波半径（主要是方框滤波聚合）

	/*
	cv::Point2d **m_ppLensCenterPoints; //每个微透镜图像中心在Raw图像中的位置
	int **m_ppPixelsMappingSet; //像素之间的映射集合，即给Raw图像上每个像素标记上标签属于第几个微透镜图像
	MatchNeighborLens ***m_ppMatchNeighborLens; //存储每一个透镜周围邻域匹配透镜的信息
	*/

	/*
	cv::Mat *m_pValidNeighborPixelsNum; //每个像素在滤波核内相邻有效像素的数目
	cv::Mat *m_pValidPixelsMask; //标记哪些像素是有效像素（微透镜区域内的像素）
	cv::Mat m_filterKnernel; //自定义滤波核
	*/
};

#endif