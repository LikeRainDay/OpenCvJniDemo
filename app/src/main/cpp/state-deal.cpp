//
// Created by iyunwen on 2017/1/12.15:19
//

#include <android/bitmap.h>
#include "state-deal-header.h"

using namespace std;

cv::Mat bitmap2Mat(JNIEnv *env, jobject bitmap);

jintArray getTypeResult(JNIEnv *env, cv::Mat mat, jint type);

JNIEXPORT jintArray  JNICALL Java_com_example_houshuai_opencvjnidemo_state_StateJni_dealStateImage
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
    switch (type) {
        case 0x1:

            break;
        case 0x2:

            break;
        case 0x3:

            break;
        default:
            break;
    }
    int size = mat.rows * mat.cols;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, (jint *) mat.data);
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
        LOGW("AndroidBitmap_getInfo() failed ! error=%d", ret);
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
    //解除锁定的像素
    AndroidBitmap_unlockPixels(env, bitmap);

    heiht = btmpInfo.height;
    width = btmpInfo.width;
    cv::Mat src(heiht, width, CV_8UC4, btmpPixels);

    if (!src.data) {
//        LOGW("bitmap failed convert to Mat return=%d!", ret);
        return nullMat;
    }
    return src;
}







