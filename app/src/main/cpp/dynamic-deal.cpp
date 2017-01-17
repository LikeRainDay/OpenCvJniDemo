//
// Created by iyunwen on 2017/1/12.15:20
//
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc.hpp>
#include "dynamic-deal-head.h"

using namespace std;

//**********************************************训练数据集的读写start****************************************************
//训练集 的读写
//class foo {
//public:
//    cv::Mat a;
//    int b;
//
//    //写入数据到XML中
//    void write(cv::FileStorage &fs) const {
//        assert(fs.isOpened());
//        fs << "{" << "a" << a << "b" << b << "}";
//    }
//
//    //读出数据从XML中
//    void read(const cv::FileNode &node) {
//        assert(node.type() == cv::FileNode::MAP);
//        node["a"] >> a;
//        node["b"] >> b;
//    }
//};
//
////写入操作
//void write(cv::FileStorage &fs, const string &, const foo &x) {
//    x.write(fs);
//}
//
////读炒作
//void read(const cv::FileNode &node, foo &x, const foo &d) {
//    if (node.empty())x = d; else x.read(node);
//}
//
//
////导入XML模板
//template<class T>
//T load_ft(const char *fname) {
//    T x;
//    cv::FileStorage f(fname, cv::FileStorage::READ);
//    f["ft object"] >> x;
//    f.release();
//    return x;
//}
//
////保存
//template<class T>
//void save_ft(const char *fname, const T &x) {
//    cv::FileStorage f(fname, cv::FileStorage::WRITE);
//    f << "ft object" << x;
//    f.release();
//}
//
//int main(){
//    foo A; save_ft<foo>("foo.xml",A);
//    foo B = load_ft<foo>("foo.xml");
//}
//**********************************************训练数据集的读写  end****************************************************

JNIEXPORT jintArray JNICALL Java_com_example_houshuai_opencvjnidemo_dynamic_DynamicJni_dynamicDetect
        (JNIEnv *env, jobject object, jbyteArray data, jint width, jint height) {
    //当前图片的数据  转化为jni的类型
    jbyte *mPicData = env->GetByteArrayElements(data, JNI_FALSE);
    if (mPicData == NULL) {
        return 0;
    }
    //将我们从Android的Java层中获取的YUV数据进行转化为RGB的颜色类型。
    cv::Mat srcImage(height, width, CV_8UC1, (unsigned char *) mPicData);
    cv::cvtColor(srcImage, srcImage, CV_YUV420p2BGRA);
    //开始进行人脸的检测  todo 动态的次要在静态的识别基础上做





















    //释放当前的jni数据
    env->ReleaseByteArrayElements(data, mPicData, 0);

}




