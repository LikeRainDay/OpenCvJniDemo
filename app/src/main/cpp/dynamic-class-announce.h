//
// Created by iyunwen on 2017/1/20.09:48
//

#ifndef OPENCVJNIDEMO_DYNAMIC_CLASS_ANNOUNCE_H
#define OPENCVJNIDEMO_DYNAMIC_CLASS_ANNOUNCE_H

#include "dynamic-deal-head.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc.hpp>
#include <iostream>

class shap_model {   //2d linear shape model
public:
    cv::Mat p; //parameter vector (kxl) cv_32f     展示表情模型时，将原始坐标投影到展示的空间里去的投影矩阵
    cv::Mat V; //linear subspace (2nxk) cv_32f     描述人脸模型的联合矩阵
    cv::Mat e; // paramter variance (kxl) cv_32f   存在联合空间内表情模型坐标的标准差矩阵
    cv::Mat C; //connectivity  (cx2) cv_32s        描述之前标注的连接关系矩阵
    /*
     * 将点集映射到合理的脸型上，它还可以为每个映射点提供单独的置信权重。
     * 这一步供人脸跟踪时调用，涉及上面Mat p矩阵，
     * */
    void calc_params(const std::vector<cv::Point2f> &pts,    //points to compute paramters
                     const cv::Mat &weight = cv::Mat(),               //weight /point (nx1) cv_32f
                     const float c_factor = 3.0);                    //clamping factor
    /**
     * 根据子空间V和方差矢量e对参数向量p进行解编码，产生新点集，
     * 这些点集在人脸跟踪和结果展示时会用到。
     * */
    std::vector<cv::Point2f>                                 //shape  described by paramters
            calc_shape();


    /**
     * 从脸型样本集中学习编码模型，每个编码模型包含相同数量的点，该函数也是本次学习的重点。
     * 下面我们先讲解普氏分析Procrustes analysis，如何注册刚性点集
     *然后讲解线性模型，如何表达局部变形。   在将Procrustes分析之前，承接上次文章，
     * 看看我们目前对什么类型的数据做操作 。
     *
     * */
    void train(
            const std::vector<std::vector<cv::Point2f>> &p, //N-example shapes
            const std::vector<cv::Vec2i> &con = std::vector<cv::Vec2i>(),//connectivity
            const float frac = 0.95,                          //fraction of variation to retain
            const int kmax = 10);                             //maximum number of modes to retain
    //..

    cv::Mat procrustes(
            const cv::Mat &x,//interleaved raw shape data columns
            const int itol,//maximum number of iterations to try
            const float ftol); //convergence tolerance
    /*函数：calc_rigid_basis，求对齐后形状空间(表情空间)的施密特标准正交基：
    *入参：矩阵X，对齐后的形状空间
    *返回值：矩阵R，n个特征点的标准正交基（R*RT=I）
    */
    cv::Mat calc_rigid_basis(const cv::Mat &X);

    /**
    * 入参：
    *    vector<vector<Point2f> >&points ,points向量存放了所有图像标注的样本点
    *返回值：
    *    Mat x， 它是一个2n*N矩阵，n表示样本点数量，N表示图像帧数
    * */
    cv::Mat pts2mat(const std::vector<std::vector<cv::Point2f> > &points);

    /**
    * Computing the in-plane rotation and scaling that
    *best aligns each shape's instance to the current estimate of the canonical shape is
    *effected through the  rot_scale_align
    * */
    cv::Mat rot_scale_align(
            const cv::Mat &src, //[x1;y1;..;xn;yn] vector of source shape
            const cv::Mat &dst  //destination shape
    );

    /**
    * 我们选用c（c=3）倍方差为阈值，对于映射后的坐标大于该阈值的，
    * 使用clamp函数进行尺寸修正。下面这段代码是在人脸跟踪时使用
    * */
    void clamp(const float c);
};

//训练集 的读写
class foo {
public:
    cv::Mat a;
    int b;

    //写入数据到XML中
    void write(cv::FileStorage &fs) const {
        assert(fs.isOpened());
        fs << "{" << "a" << a << "b" << b << "}";
    }

    //读出数据从XML中
    void read(const cv::FileNode &node) {
        assert(node.type() == cv::FileNode::MAP);
        node["a"] >> a;
        node["b"] >> b;
    }
};

//导入XML模板
template<class T>
T load_ft(const char *fname) {
    T x;
    cv::FileStorage f(fname, cv::FileStorage::READ);
    f["ft object"] >> x;
    f.release();
    return x;
}

//保存
template<class T>
void save_ft(const char *fname, const T &x) {
    cv::FileStorage f(fname, cv::FileStorage::WRITE);
    f << "ft object" << x;
    f.release();
}


class path_model {
public:
    cv::Mat P; //normalized patch
    //...
    cv::Mat //response map
            calc_response(
            const cv::Mat &im, //image patch of search region
            const bool sum2one = false); //normalize to sum-to-one?
    // ...

    /**
     * 输入参数含义：
     * images：包含多个样本图像的矩阵向量（原始含有人像的图像）
     * psize：团块模型窗口的大小
     * var：手工标注错误的方差（生成理想图像时使用）
     * lambda：调整的参数（调整上一次得到的团块模型的大小，以便于当前目标函数偏导数作差）
     * mu_init：初始步长（构造梯度下降法求团块模型时的更新速率）
     * nsamples：随机选取的样本数量（梯度下降算法迭代的次数）
     * visi：训练过程是否可观察标志
     *
     * */
    void train(const std::vector<cv::Mat> &images, //training image patches
               const cv::Size psize, //patch size
               const float var = 1.0, //ideal response variance
               const float lambda = 1e-6, //regularization weight
               const float mu_init = 1e-3, //initial step size
               const int nsamples = 1000, //number of samples
               const bool visi = false); //visualize process?
    //...

    /**
     * 求灰度图
     * */
    cv::Mat convert_image(const cv::Mat &im);

    /**
     *对单幅样本图像仿射变换矩阵的计算并没有迭代，采用更加简单的方式：
     *(1)   计算手工标注点的重心
     *(2)   对手工标注点去中心化
     *(3)   利用最小二乘法计算样本点到参考坐标的旋转角度
     * */
    cv::Mat calc_simil(const cv::Mat &pts);


};


#endif //OPENCVJNIDEMO_DYNAMIC_CLASS_ANNOUNCE_H
