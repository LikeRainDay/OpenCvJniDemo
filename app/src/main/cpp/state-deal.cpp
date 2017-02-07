//
// Created by iyunwen on 2017/1/12.15:19
//

#include <android/bitmap.h>
#include "state-deal-header.h"

using namespace std;
bool isDetected = false;

cv::Mat bitmap2Mat(JNIEnv *env, jobject bitmap);

void detectAndDraw(cv::Mat &img,
                   cv::CascadeClassifier &cascade,
                   double scale);

jintArray getTypeResult(JNIEnv *env, cv::Mat mat, jint type);

JNIEXPORT jintArray  JNICALL Java_com_example_houshuai_opencvjnidemo_bitmap_1state_BitmapStateJni_dealStateImage
        (JNIEnv *env, jobject object, jobject bitmap, jint type) {
    //结果数组
    jintArray result;
    //获得转换后的Bitmap
    cv::Mat src = bitmap2Mat(env, bitmap);
    //如果当前的Mat为空 则返回
    if (src.empty()) {
        return NULL;
    }
    //转化为结构 输出
    result = getTypeResult(env, src, type);
    //返回图片数组
    return result;
}


/**
 * 进行赛选并返回处理结果的Jintarray
 * */

jintArray getTypeResult(JNIEnv *env, cv::Mat mat, jint type) {
    int size = mat.rows * mat.cols;
    jintArray result = env->NewIntArray(size);

    switch (type) {
        case 0x1:  //原图进行复现
            env->SetIntArrayRegion(result, 0, size, (const jint *) mat.data);
            break;
        case 0x2:  //进行转化为灰度图
            cv::cvtColor(mat, mat, cv::COLOR_BGRA2GRAY, 1);     //将其转化为灰度图
            env->SetIntArrayRegion(result, 0, size, (const jint *) mat.data);
            break;
        case 0x3:  //进行高斯模糊
            cvSmooth(&mat, &mat, CV_GAUSSIAN, 11, 0, 0, 0);
            env->SetIntArrayRegion(result, 0, size, (const jint *) mat.data);
            break;
        case 0x4:  //进行人脸的刚性追踪




            break;
        default:
            break;
    }
    return result;
}


/**
 * 将Bitmap转化为
 * */
cv::Mat bitmap2Mat(JNIEnv *env, jobject bitmap) {
    AndroidBitmapInfo btmpInfo;         //bitmap的信息
    void *btmpPixels;                   //bitmap的像素信息
    int heiht, width, ret, y, x;
    cv::Mat nullMat;

    //解析当前的Bitmap
    if ((ret = AndroidBitmap_getInfo(env, bitmap, &btmpInfo)) < 0) {
//        LOGW("AndroidBitmap_getInfo() failed ! error=%d", ret);
        return nullMat;
    }
    /**
     * RGBA_8888格式的Bitmap，一个像素占32位，分别是A：8bit，R：8bit；G：8bit；B：8bit。
     * 对应到opencv的Mat对象，有个一个Mat的构造函数，结果接受外部数据数组并生Mat对象，
     * 这将大大方便转换过程，不用逐像素的操作了
     *
     * 因为RGBA_8888格式的存储方式正好与Mat的CV_8UC4对应
     * */
    if (btmpInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
//        LOGW("Bitmap format is not RGBA_8888!");
        return nullMat;
    }

    //获取当前Bitmap中的像素
    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &btmpPixels)) < 0) {
//        LOGW("First Bitmap LockPixels Failed return=%d!", ret);
        return nullMat;
    }


    heiht = btmpInfo.height;
    width = btmpInfo.width;
    cv::Mat src(heiht, width, CV_8UC4, btmpPixels);
    cv::Mat dst = src.clone();
    // init our output image
    float alpha = 1.9;
    float beta = -80;
    dst.convertTo(dst, -1, alpha, beta);
    //解除锁定的像素
    AndroidBitmap_unlockPixels(env, bitmap);

    if (!dst.data) {
//        LOGW("bitmap failed convert to Mat return=%d!", ret);
        return nullMat;
    }
    return dst;
}


/**
 * 用于进行脸部高斯模糊
 * */
void detectAndDraw(cv::Mat &img,
                   cv::CascadeClassifier &cascade,
                   double scale) {
    int i = 0;
    double t = 0;
    vector<cv::Rect> faces;
    const static cv::Scalar colors[] = {CV_RGB(0, 0, 255),
                                        CV_RGB(0, 128, 255),
                                        CV_RGB(0, 255, 255),
                                        CV_RGB(0, 255, 0),
                                        CV_RGB(255, 128, 0),
                                        CV_RGB(255, 255, 0),
                                        CV_RGB(255, 0, 0),
                                        CV_RGB(255, 0, 255)};//用不同的颜色表示不同的人脸

    cv::Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale),
                           CV_8UC1);//将图片缩小，加快检测速度

    cv::cvtColor(img, gray, CV_BGR2GRAY);//因为用的是类haar特征，所以都是基于灰度图像的，这里要转换成灰度图像
    cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);//将尺寸缩小到1/scale,用线性插值
    cv::equalizeHist(smallImg, smallImg);//直方图均衡




    cascade.detectMultiScale(smallImg, faces,
                             1.1, 2, 0
                                     //|CV_HAAR_FIND_BIGGEST_OBJECT
                                     //|CV_HAAR_DO_ROUGH_SEARCH
                                     | CV_HAAR_SCALE_IMAGE,
                             cv::Size(30, 30));

    for (vector<cv::Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++) {
        isDetected = true;
        cv::Mat smallImgROI;
        vector<cv::Rect> nestedObjects;
        cv::Point center, left, right;
        cv::Scalar color = colors[i % 8];
        int radius;
        center.x = cvRound((r->x + r->width * 0.5) * scale);//还原成原来的大小
        center.y = cvRound((r->y + r->height * 0.5) * scale);
        radius = cvRound((r->width + r->height) * 0.25 * scale);


        left.x = center.x - radius;
        left.y = cvRound(center.y - radius * 1.3);

        if (left.y < 0) {
            left.y = 0;
        }

        right.x = center.x + radius;
        right.y = cvRound(center.y + radius * 1.3);

        if (right.y > img.rows) {
            right.y = img.rows;
        }

        cv::rectangle(img, left, right, cv::Scalar(255, 0, 0));


        cv::Mat roi = img(cv::Range(left.y, right.y), cv::Range(left.x, right.x));
        cv::Mat dst;

        int value1 = 3, value2 = 1;

        int dx = value1 * 5;    //双边滤波参数之一
        double fc = value1 * 12.5; //双边滤波参数之一
        int p = 50;//透明度
        cv::Mat temp1, temp2, temp3, temp4;

        //双边滤波
        cv::bilateralFilter(roi, temp1, dx, fc, fc);

        temp2 = (temp1 - roi + 128);

        //高斯模糊
        cv::GaussianBlur(temp2, temp3, cv::Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

        temp4 = roi + 2 * temp3 - 255;

        dst = (roi * (100 - p) + temp4 * p) / 100;


        dst.copyTo(roi);
    }
}




