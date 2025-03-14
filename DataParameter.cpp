#include "DataParameter.h"
using namespace cv;

using namespace std;

DataParameter::DataParameter()
{
	m_rawImageParameter.m_yCenterBeginOffset = 2;
	m_rawImageParameter.m_xCenterBeginOffset = 2;
	m_rawImageParameter.m_yCenterEndOffset = 2;
	m_rawImageParameter.m_xCenterEndOffset = 2;
	m_filterRadius = 4;
	m_microImageParameter.m_circleDiameter = 34.0;
	m_microImageParameter.m_circleNarrow = 1.5;
	m_disparityParameter.m_dispMin = 5;
	m_disparityParameter.m_dispMax = 13;
	m_disparityParameter.m_dispStep = 0.5;
	m_microImageParameter.m_ppLensCenterPoints = nullptr;
	m_microImageParameter.m_ppPixelsMappingSet = nullptr;
	m_microImageParameter.m_ppMatchNeighborLens = nullptr;
	m_filterPatameter.m_pValidNeighborPixelsNum = nullptr;
	m_filterPatameter.m_pValidPixelsMask = nullptr;
}

DataParameter::DataParameter(std::string dataFolderName, std::string centerPointFileName, std::string inputImgName,
	int yCenterBeginOffset, int xCenterBeginOffset, int yCenterEndOffset, int xCenterEndOffset,
	int filterRadius, float circleDiameter, float circleNarrow, int dispMin, int dispMax, float dispStep)
{
	init(dataFolderName, centerPointFileName, inputImgName, yCenterBeginOffset, xCenterBeginOffset, yCenterEndOffset, xCenterEndOffset,
		filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);
}

DataParameter::~DataParameter()//�����������������ڽ������Զ�����
{
	if (m_microImageParameter.m_ppLensCenterPoints)
	{
		for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
			delete[]m_microImageParameter.m_ppLensCenterPoints[y];
		delete[]m_microImageParameter.m_ppLensCenterPoints;
	}
	if (m_microImageParameter.m_ppPixelsMappingSet)
	{
		for (int y = 0; y < m_rawImageParameter.m_srcImgHeight; y++)
			delete[]m_microImageParameter.m_ppPixelsMappingSet[y];
		delete[]m_microImageParameter.m_ppPixelsMappingSet;
	}
	if (m_microImageParameter.m_ppMatchNeighborLens)
	{
		for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
			delete[]m_microImageParameter.m_ppMatchNeighborLens[y];
		delete[]m_microImageParameter.m_ppMatchNeighborLens;
	}
	if (m_filterPatameter.m_pValidPixelsMask)
	{
		delete m_filterPatameter.m_pValidPixelsMask;
	}
	if (m_filterPatameter.m_pValidNeighborPixelsNum)
	{
		delete m_filterPatameter.m_pValidNeighborPixelsNum;
	}
}

void DataParameter::init(std::string dataFolderName, std::string centerPointFileName, std::string inputImgName,
	int yCenterBeginOffset, int xCenterBeginOffset, int yCenterEndOffset, int xCenterEndOffset,
	int filterRadius, float circleDiameter, float circleNarrow, int dispMin, int dispMax, float dispStep)
{
	lensCenterPointsInit(dataFolderName, centerPointFileName);
	validLensCenterInit(yCenterBeginOffset, xCenterBeginOffset, yCenterEndOffset, xCenterEndOffset);
	imageBaseMessageInit(inputImgName, filterRadius, circleDiameter, circleNarrow, dispMin, dispMax, dispStep);
	generatePixelsMappingSet();
	generateNeighborCenterPoints();
	generateValidPoints();

	std::cout << "DataParameter::init final!" << std::endl;
}

void DataParameter::lensCenterPointsInit(std::string dataFolderPath, std::string centerPointFileName)
{//͸�����ĵ��ʼ��
	m_folderPath = dataFolderPath;
	std::string centerFileTxtName = dataFolderPath + "/" + centerPointFileName;
	std::ifstream ifs;
	ifs.open(centerFileTxtName, std::ifstream::in);
	ifs >> m_rawImageParameter.m_yLensNum >> m_rawImageParameter.m_xLensNum;//���ļ��ж�ȡ͸�������� m_yLensNum ������ m_xLensNum�����洢�� m_rawImageParameter �ṹ���С�
	m_microImageParameter.m_ppLensCenterPoints = new Point2d *[m_rawImageParameter.m_yLensNum];
	for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
		m_microImageParameter.m_ppLensCenterPoints[y] = new Point2d[m_rawImageParameter.m_xLensNum];

	for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
		for (int x = 0; x < m_rawImageParameter.m_xLensNum; x++)
			ifs >> m_microImageParameter.m_ppLensCenterPoints[y][x].y >> m_microImageParameter.m_ppLensCenterPoints[y][x].x;

	ifs.close();
	float x1 = m_microImageParameter.m_ppLensCenterPoints[0][0].x; //��0����ߵ�һ��x����
	float x2 = m_microImageParameter.m_ppLensCenterPoints[1][0].x; //��1����ߵ�һ��x����

	if (x1 > x2) m_lensArrageMode = 0; //��0�е�һ��Բ������ 
	else m_lensArrageMode = 1; //��0�е�һ��Բ����͹��
	std::cout << "lensCenterPointsInit final!" << std::endl;
}

void DataParameter::validLensCenterInit(int yCenterBeginOffset, int xCenterBeginOffset, int yCenterEndOffset, int xCenterEndOffset)
{//�����Щ͸��������Ч
	m_rawImageParameter.m_yCenterBeginOffset = yCenterBeginOffset;
	m_rawImageParameter.m_xCenterBeginOffset = xCenterBeginOffset;
	m_rawImageParameter.m_yCenterEndOffset = yCenterEndOffset;
	m_rawImageParameter.m_xCenterEndOffset = xCenterEndOffset;

	Point2d &topLeftCenterPos = m_microImageParameter.m_ppLensCenterPoints[yCenterBeginOffset - 1][xCenterBeginOffset - 1];
	Point2d &topRightCenterPos = m_microImageParameter.m_ppLensCenterPoints[yCenterBeginOffset - 1][m_rawImageParameter.m_xLensNum - xCenterEndOffset];
	Point2d &belowLeftCenterPos = m_microImageParameter.m_ppLensCenterPoints[m_rawImageParameter.m_yLensNum - yCenterEndOffset][xCenterBeginOffset - 1];
	Point2d &belowRightCenterPos = m_microImageParameter.m_ppLensCenterPoints[m_rawImageParameter.m_yLensNum - yCenterEndOffset][m_rawImageParameter.m_xLensNum - xCenterEndOffset];

	double left = std::min(topLeftCenterPos.x, belowLeftCenterPos.x);
	double right = std::max(topRightCenterPos.x, belowRightCenterPos.x);
	double top = std::min(topLeftCenterPos.y, topRightCenterPos.y);
	double below = std::max(belowLeftCenterPos.y, belowRightCenterPos.y);

	m_rawImageParameter.m_yPixelBeginOffset = top;
	m_rawImageParameter.m_xPixelBeginOffset = left;

	m_rawImageParameter.m_recImgHeight = below - top + 1;
	m_rawImageParameter.m_recImgWidth = right - left + 1;
	cout<<"validLensCenterInit_final"<<endl;
}

void DataParameter::imageBaseMessageInit(std::string inputImgName, int filterRadius, float circleDiameter,
	float circleNarrow, int dispMin, int dispMax, float dispStep)
{//ͼ�������Ϣ���Ӳ���Ϣ��ʼ��
	cout<<"imageBaseMessageInit_start"<<endl;
	m_filterRadius = filterRadius;//�˲��뾶
	m_microImageParameter.m_circleDiameter = circleDiameter;//԰ֱ��
	m_microImageParameter.m_circleNarrow = circleNarrow;//Բ�뾶��Сֵ
	m_disparityParameter.m_dispMin = dispMin;
	m_disparityParameter.m_dispMax = dispMax;
	m_disparityParameter.m_dispStep = dispStep;
	m_disparityParameter.m_disNum = double(dispMax - dispMin) / m_disparityParameter.m_dispStep; //�Ӳ�label��Ŀ
	m_filterPatameter.m_filterKnernel = cv::Mat::ones(2 * filterRadius + 1, 2 * filterRadius + 1, CV_32FC1);//�����˲��뾶���������
	m_microImageParameter.m_radiusDisEqu = (circleDiameter / 2 - m_microImageParameter.m_circleNarrow)*(circleDiameter / 2 - m_microImageParameter.m_circleNarrow);

	std::string inputImagePath = m_folderPath + "/" + inputImgName;
	cout<<"m_folderPath："<<m_folderPath<<endl;
	cout<<"inputImagePath："<<inputImagePath<<endl;
	
	m_inputImg = imread(inputImagePath, IMREAD_COLOR);
	
	//m_inputImg = imread(inputImagePath, IMREAD_GRAYSCALE);
	m_rawImageParameter.m_srcImgWidth = m_inputImg.cols;
	m_rawImageParameter.m_srcImgHeight = m_inputImg.rows;


	m_inputImgRec = m_inputImg(cv::Rect(m_rawImageParameter.m_xPixelBeginOffset, m_rawImageParameter.m_yPixelBeginOffset, 
		m_rawImageParameter.m_recImgWidth, m_rawImageParameter.m_recImgHeight)).clone();

	std::string recImageStore = m_folderPath + "/" + "srcImgRec.png";
	imwrite(recImageStore, m_inputImgRec);
	cout<<"imageBaseMessageInit_final"<<endl;
}

void DataParameter::generatePixelsMappingSet()
{//�������ص�ӳ�伯��
	m_microImageParameter.m_ppPixelsMappingSet = new int *[m_rawImageParameter.m_srcImgHeight];
	for (int y = 0; y < m_rawImageParameter.m_srcImgHeight; y++)
	{
		m_microImageParameter.m_ppPixelsMappingSet[y] = new int[m_rawImageParameter.m_srcImgWidth];
		memset(m_microImageParameter.m_ppPixelsMappingSet[y], -1, m_rawImageParameter.m_srcImgWidth*sizeof(int));
	}

	//�����Щ��������Ч��
	m_filterPatameter.m_pValidPixelsMask = new cv::Mat;
	*m_filterPatameter.m_pValidPixelsMask = Mat::zeros(m_rawImageParameter.m_srcImgHeight, m_rawImageParameter.m_srcImgWidth, CV_32FC1);
//#pragma omp parallel for 
	for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
		for (int x = 0; x < m_rawImageParameter.m_xLensNum; x++)
			generatePixelsMappingSet(y, x);

	std::cout << "generatePixelsMappingSet final!" << std::endl;
}

void DataParameter::generatePixelsMappingSet(int y, int x)
{//�������ص�ӳ�伯��--����ÿ����ͼ���cost
	Point2d &centerPos = m_microImageParameter.m_ppLensCenterPoints[y][x];
	//������ǰ͸������λ�ø���������
	for (int py = centerPos.y - m_microImageParameter.m_circleDiameter / 2 + m_microImageParameter.m_circleNarrow; 
		py <= centerPos.y + m_microImageParameter.m_circleDiameter / 2 - m_microImageParameter.m_circleNarrow; py++)
	{
		//�����������Ϊ1������ô�����������Ч
		float *yDataRowsMask = (float *)(*m_filterPatameter.m_pValidPixelsMask).ptr<float *>(py);
		for (int px = centerPos.x - m_microImageParameter.m_circleDiameter / 2 + m_microImageParameter.m_circleNarrow; 
			px <= centerPos.x + m_microImageParameter.m_circleDiameter / 2 - m_microImageParameter.m_circleNarrow; px++)
		{
			//΢͸��������
			int lens_num = y*m_rawImageParameter.m_xLensNum + x;
			//��������Ƿ���Բ��
			if ((centerPos.y - py)*(centerPos.y - py) + (centerPos.x - px)*(centerPos.x - px) <= m_microImageParameter.m_radiusDisEqu)
			{
				m_microImageParameter.m_ppPixelsMappingSet[py][px] = lens_num;
				yDataRowsMask[px] = 1.0;
			}
		}
	}
}

void DataParameter::generateNeighborCenterPoints()
{//����ÿ�����ĵ���Χ�Ķ�Ӧ���ĵ�
	//ÿ��͸�����ĵ���ھ����ĵ���Ϣ
	m_microImageParameter.m_ppMatchNeighborLens = new MatchNeighborLens **[m_rawImageParameter.m_yLensNum];
	for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
	{
		m_microImageParameter.m_ppMatchNeighborLens[y] = new MatchNeighborLens *[m_rawImageParameter.m_xLensNum];
		for (int x = 0; x < m_rawImageParameter.m_xLensNum; x++)
			m_microImageParameter.m_ppMatchNeighborLens[y][x] = new MatchNeighborLens[NEIGHBOR_MATCH_LENS_NUM];
	}

//#pragma omp parallel for 
	for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
		for (int x = 0; x < m_rawImageParameter.m_xLensNum; x++)
			generateNeighborCenterPoints(y, x);

	std::cout << "generateNeighborCenterPoints final!" << std::endl;
}

void DataParameter::generateNeighborCenterPoints(int y, int x)
{//����ÿ�����ĵ���Χ�Ķ�Ӧ���ĵ�
	int parityFlag = y & 1;//������Ϊ1��ż����Ϊ0
	int x_shift = 0;
	if (m_lensArrageMode ^ parityFlag){
		//��͹��ż���У������󰼣�������
		x_shift = -1;
	}
	else{
		//��͹�������У������󰼣�ż����
		x_shift = 1;
	}

	int numCount = 0;
	Point2d &curCenterPoint = m_microImageParameter.m_ppLensCenterPoints[y][x];
	double cy, cx, dis;

	
	if (y - 1 >= 0){//�Ϸ��ھӣ��൱���������������������������
		cy = m_microImageParameter.m_ppLensCenterPoints[y - 1][x].y;
		cx = m_microImageParameter.m_ppLensCenterPoints[y - 1][x].x;
		dis = sqrt((cy - curCenterPoint.y)*(cy - curCenterPoint.y) + (cx - curCenterPoint.x)*(cx - curCenterPoint.x));
		m_microImageParameter.m_ppMatchNeighborLens[y][x][numCount++] = MatchNeighborLens(cy, cx, dis, (y - 1)*m_rawImageParameter.m_xLensNum + x);

		if ((x_shift < 0 && x - 1 >= 0) || (x_shift > 0 && x + 1 < m_rawImageParameter.m_xLensNum)){
			cy = m_microImageParameter.m_ppLensCenterPoints[y - 1][x + x_shift].y;
			cx = m_microImageParameter.m_ppLensCenterPoints[y - 1][x + x_shift].x;
			dis = sqrt((cy - curCenterPoint.y)*(cy - curCenterPoint.y) + (cx - curCenterPoint.x)*(cx - curCenterPoint.x));
			m_microImageParameter.m_ppMatchNeighborLens[y][x][numCount++] = MatchNeighborLens(cy, cx, dis, (y - 1)*m_rawImageParameter.m_xLensNum + x + x_shift);
		}
	}

	if (y + 1 < m_rawImageParameter.m_yLensNum){//�·��ھ�
		cy = m_microImageParameter.m_ppLensCenterPoints[y + 1][x].y;
		cx = m_microImageParameter.m_ppLensCenterPoints[y + 1][x].x;
		dis = sqrt((cy - curCenterPoint.y)*(cy - curCenterPoint.y) + (cx - curCenterPoint.x)*(cx - curCenterPoint.x));
		m_microImageParameter.m_ppMatchNeighborLens[y][x][numCount++] = MatchNeighborLens(cy, cx, dis, (y + 1)*m_rawImageParameter.m_xLensNum + x);

		if ((x_shift < 0 && x - 1 >= 0) || (x_shift > 0 && x + 1 < m_rawImageParameter.m_xLensNum)){
			cy = m_microImageParameter.m_ppLensCenterPoints[y + 1][x + x_shift].y;
			cx = m_microImageParameter.m_ppLensCenterPoints[y + 1][x + x_shift].x;
			dis = sqrt((cy - curCenterPoint.y)*(cy - curCenterPoint.y) + (cx - curCenterPoint.x)*(cx - curCenterPoint.x));
			m_microImageParameter.m_ppMatchNeighborLens[y][x][numCount++] = MatchNeighborLens(cy, cx, dis, (y + 1)*m_rawImageParameter.m_xLensNum + x + x_shift);
		}
	}

	if (x - 1 >= 0){//����ھ�
		cy = m_microImageParameter.m_ppLensCenterPoints[y][x - 1].y;
		cx = m_microImageParameter.m_ppLensCenterPoints[y][x - 1].x;
		dis = sqrt((cy - curCenterPoint.y)*(cy - curCenterPoint.y) + (cx - curCenterPoint.x)*(cx - curCenterPoint.x));
		m_microImageParameter.m_ppMatchNeighborLens[y][x][numCount++] = MatchNeighborLens(cy, cx, dis, y*m_rawImageParameter.m_xLensNum + x - 1);
	}

	if (x + 1 < m_rawImageParameter.m_xLensNum){//�Ҳ��ھ�
		cy = m_microImageParameter.m_ppLensCenterPoints[y][x + 1].y;
		cx = m_microImageParameter.m_ppLensCenterPoints[y][x + 1].x;
		dis = sqrt((cy - curCenterPoint.y)*(cy - curCenterPoint.y) + (cx - curCenterPoint.x)*(cx - curCenterPoint.x));
		m_microImageParameter.m_ppMatchNeighborLens[y][x][numCount++] = MatchNeighborLens(cy, cx, dis, y*m_rawImageParameter.m_xLensNum + x + 1);
	}

	if (numCount < NEIGHBOR_MATCH_LENS_NUM)//�����������Ϣ
		m_microImageParameter.m_ppMatchNeighborLens[y][x][numCount++] = MatchNeighborLens(-1, -1, -1, -1);
	
}

void DataParameter::generateValidPoints()
{//����ÿ��СԲ�����أ�����Χ��Ч�����
	m_filterPatameter.m_pValidNeighborPixelsNum = new cv::Mat;
	*m_filterPatameter.m_pValidNeighborPixelsNum = cv::Mat::zeros(m_rawImageParameter.m_srcImgHeight, m_rawImageParameter.m_srcImgWidth, CV_32FC1);

//#pragma omp parallel for
	for (int y = 0; y < m_rawImageParameter.m_yLensNum; y++)
		for (int x = 0; x < m_rawImageParameter.m_xLensNum; x++)
			generateValidPoints(y, x);

	validPointsBoundaryRepair();
	std::cout << "generateValidPoints final!" << std::endl;
}

void DataParameter::generateValidPoints(int y, int x)
{//����ÿ��СԲ�����أ�����Χ��Ч�����
	Point2d &curCenterPos = m_microImageParameter.m_ppLensCenterPoints[y][x];
	int x_begin = curCenterPos.x - m_microImageParameter.m_circleDiameter / 2 + m_microImageParameter.m_circleNarrow;
	int y_begin = curCenterPos.y - m_microImageParameter.m_circleDiameter / 2 + m_microImageParameter.m_circleNarrow;
	int x_end = curCenterPos.x + m_microImageParameter.m_circleDiameter / 2 - m_microImageParameter.m_circleNarrow;
	int y_end = curCenterPos.y + m_microImageParameter.m_circleDiameter / 2 - m_microImageParameter.m_circleNarrow;

	//��ȡСԲ����
	cv::Mat srcCost = (*m_filterPatameter.m_pValidPixelsMask)(cv::Rect(x_begin, y_begin, x_end - x_begin + 1, y_end - y_begin + 1));
	cv::Mat destCost = (*m_filterPatameter.m_pValidNeighborPixelsNum)(cv::Rect(x_begin, y_begin, x_end - x_begin + 1, y_end - y_begin + 1));



	cv::filter2D(srcCost, destCost, -1, m_filterPatameter.m_filterKnernel, cv::Point(-1, -1), 0, BORDER_CONSTANT);
	destCost = destCost.mul(srcCost);//������ЧԪ�أ�ͳ����Ч�ھ���
}

void DataParameter::validPointsBoundaryRepair()
{//����Ч��ı߽��������ؽ�����1��������ֹ���ֳ�0����
//#pragma omp parallel for
	for (int py = 0; py < m_rawImageParameter.m_srcImgHeight; py++)
	{
		float *yDataRows = (float *)(*m_filterPatameter.m_pValidNeighborPixelsNum).ptr<float *>(py);
		for (int px = 0; px < m_rawImageParameter.m_srcImgWidth; px++)
		{
			if (yDataRows[px] < 0.1) // �������Ч�����ٻ� >=1
				yDataRows[px] = 1.0;
		}
	}
}