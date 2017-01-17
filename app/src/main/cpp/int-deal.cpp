//
// Created by iyunwen on 2017/1/17.10:33
//

#include "int-deal-head.h"


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
        }
        case 3://高斯模糊
        {
            cv::Mat src(h,w, CV_8UC4, (unsigned char *) srcData);
            cvSmooth(&src, &src, CV_GAUSSIAN, 11, 0, 0, 0);
            env->SetIntArrayRegion(result, 0, size, (const jint *) src.data);
            break;
        }

        case 4://非刚性人脸追踪
        {

            cv::Mat src(h,w, CV_8UC4, (unsigned char *) srcData);
            cv::cvtColor(src, src, cv::COLOR_BGRA2GRAY, 0);
            env->SetIntArrayRegion(result, 0, size, (const jint *) src.data);


            break;
        }
        default:
            break;
    }
    //进行释放当前的类型
    env->ReleaseIntArrayElements(data, srcData, 0);
    return result;

}
