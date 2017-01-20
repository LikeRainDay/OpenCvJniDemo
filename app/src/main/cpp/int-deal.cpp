//
// Created by iyunwen on 2017/1/17.10:33
//

#include "int-deal-head.h"
#include <vector>

using namespace cv;
using namespace std;

//IplImage *change4channelTo3InIplImage(IplImage *src);

/**
 * 元数据类
 * */
class ft_data {
public:
    vector<int> symmetry;
    vector<Vec2i> connections;
    vector<Mat> imnames;    //用于存放当前的Mat
    vector<vector<Point2f>> points;

    //.......
    Mat get_image(const int idx, const int flag);

    vector<Point2f> get_points(const int idx, const bool flipped);
};


JNIEXPORT
jintArray JNICALL Java_com_example_houshuai_opencvjnidemo_int_1state_IntJni_intDetect
        (JNIEnv *env, jobject object, jintArray data, jint w, jint h, jint type) {
    //转化为JNi调用的类型
    jint *srcData;
    srcData = env->GetIntArrayElements(data, JNI_FALSE);
    if (NULL == srcData) {
        return 0;
    }
    int size = w * h;
    jintArray result = env->NewIntArray(size);
    switch (type) {
        case 1://原图展示
        {
            env->SetIntArrayRegion(result, 0, size, srcData);
            break;
        }
        case 2://灰度图
        {
            int alpha = 0xFF << 24;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    // 获得像素的颜色
                    int color = srcData[w * i + j];
                    int red = ((color & 0x00FF0000) >> 16);
                    int green = ((color & 0x0000FF00) >> 8);
                    int blue = color & 0x000000FF;
                    color = (red + green + blue) / 3;
                    color = alpha | (color << 16) | (color << 8) | color;
                    srcData[w * i + j] = color;
                }
            }
            env->SetIntArrayRegion(result, 0, size, srcData);
            break;
//            Mat src(h, w, CV_8UC4, srcData);
////            cvCvtColor(&src, &src, cv::COLOR_BGRA2GRAY);
//            uchar *ptr = src.ptr(0);
//            for (int i = 0; i < w * h; i++) {
//                int grayScale = (int) (ptr[4 * i + 2] * 0.299 + ptr[4 * i + 1] * 0.587
//                                       + ptr[4 * i + 0] * 0.114);
//                ptr[4 * i + 1] = (uchar) grayScale;
//                ptr[4 * i + 2] = (uchar) grayScale;
//                ptr[4 * i + 0] = (uchar) grayScale;
//            }
//            env->SetIntArrayRegion(result, 0, size, srcData);

        }
        case 3://高斯模糊
        {
            Mat src(h, w, CV_8UC4, (unsigned char *) srcData);
            cvtColor(src, src, COLOR_BGRA2GRAY);
            env->SetIntArrayRegion(result, 0, size, (const jint *) src.data);


            break;
        }

        case 4://非刚性人脸追踪
        {
//            Mat myimg(h, w, CV_8UC4, (unsigned char *) srcData);
//            IplImage image = IplImage(myimg);
//            IplImage *image3channel = change4channelTo3InIplImage(&image);
//            IplImage *pCannyImage = cvCreateImage(cvGetSize(image3channel), IPL_DEPTH_8U, 1);
//            cvCanny(image3channel, pCannyImage, 50, 150, 3);
//            int *outImage = new int[w * h];
//            for (int i = 0; i < w * h; i++) {
//                outImage[i] = (int) pCannyImage->imageData[i];
//            }
//            env->SetIntArrayRegion(result, 0, size, outImage);












            break;
        }
        default:
            break;
    }
    //进行释放当前的类型
    env->ReleaseIntArrayElements(data, srcData, 0);
    return result;

}





//
//IplImage *change4channelTo3InIplImage(IplImage *src) {
//    if (src->nChannels != 4) {
//        return NULL;
//    }
//    IplImage *destImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3);
//    for (int row = 0; row < src->height; row++) {
//        for (int col = 0; col < src->width; col++) {
//            CvScalar s = cvGet2D(src, row, col);
//            cvSet2D(destImg, row, col, s);
//        }
//    }
//
//    return destImg;
//}