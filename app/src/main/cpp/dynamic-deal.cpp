//
// Created by iyunwen on 2017/1/12.15:20
//

#include "dynamic-class-announce.h"


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
        (JNIEnv *env, jobject, jbyteArray data, jint width, jint height) {
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


void shap_model::calc_params(const std::vector<cv::Point2f> &pts, const cv::Mat &weight,
                             const float c_factor) {
    int n = pts.size();
    assert(V.rows == 2 * n);
    cv::Mat s = Mat(pts).reshape(1, 2 * n); //point set to vector format
    if (weight.empty())p = V.t() * s;   //simple projection
    else {                            //scaled projection
        if (weight.rows != n) {
            std::cout << "Invalid weighting matrix" << std::endl;
            abort();
        }
        int K = V.cols;
        cv::Mat H = cv::Mat::zeros(K, K, CV_32F), g = cv::Mat::zeros(K, 1, CV_32F);
        for (int i = 0; i < n; i++) {
            cv::Mat v = V(cv::Rect(0, 2 * i, K, 2));
            float w = weight.at(i);
            H += w * v.t() * v;
            g += w * v.t() * Mat(pts[i]);
        }
        solve(H, g, p, cv::DECOMP_SVD);
    }
    this->clamp(c_factor);          //clamp resulting parameters

}


std::vector<cv::Point2f> shap_model::calc_shape() {
    cv::Mat s = V * p;
    int n = s.rows / 2;
    std::vector<cv::Point2f> pts;
    for (int i = 0; i < n; i++)pts.push_back(cv::Point2f(s.at(2 * i), s.at(2 * i + 1)));
    return pts;
}

void shap_model::train(const std::vector<std::vector<cv::Point2f>> &p,
                       const std::vector<cv::Vec2i> &con,
                       const float frac, const int kmax) {
    //vectorize points
    cv::Mat X = this->pts2mat(p);
    int N = X.cols, n = X.rows / 2;

    //align shapes
    cv::Mat Y = this->procrustes(X);

    //compute rigid transformation, rigid subspace
    cv::Mat R = this->calc_rigid_basis(Y);

    //compute non-rigid transformation
    cv::Mat P = R.t() * Y;
    cv::Mat dY = Y - R * P; //project-out rigidity

    /*Data = U*w*Vt ，奇异值矩阵为w*/
    cv::SVD svd(dY * dY.t());
    int m = std::min(std::min(kmax, N - 1), n - 1);
    float vsum = 0;
    for (int i = 0; i < m; i++)
        vsum += svd.w.at(i);

    float v = 0;
    int k = 0;
    for (k = 0; k < m; k++) {
        v += svd.w.at(k);
        if (v / vsum >= frac) {
            k++;
            break;
        }
    }  /*取前k个奇异值*/
    if (k > m)
        k = m;
    cv::Mat D = svd.u(cv::Rect(0, 0, k, 2 * n)); /*非刚性变化投影*/
    //combine bases

    V.create(2 * n, 4 + k, CV_32F);
    cv::Mat Vr = V(cv::Rect(0, 0, 4, 2 * n));
    R.copyTo(Vr);//rigid subspace
    cv::Mat Vd = V(cv::Rect(4, 0, k, 2 * n));
    D.copyTo(Vd);//nonrigid subspace

    //compute variance (normalized wrt scale)
    cv::Mat Q = V.t() * X; //矩阵Q即为联合分布空间中的新坐标集合
    for (int i = 0; i < N; i++) {
        /*用Q的第一行元素分别去除对应的0~K+4行元素，归一化新空间的scale，
         *防止数据样本（联合分布投影后）的相对尺度过大，影响后面的判断
        */
        float v = Q.at(0, i);
        cv::Mat q = Q.col(i);
        q /= v;
    }
    e.create(4 + k, 1, CV_32F);
    /*为了计算方差*/
    pow(Q, 2, Q);
    for (int i = 0; i < 4 + k; i++) {
        if (i < 4)
            e.at(i) = -1;     //no clamping for rigid coefficients，矩阵Q的前4列为刚性系数
        else
            e.at(i) = (int) (Q.row(i).dot(cv::Mat::ones(1, N, CV_32F)) /
                             (N - 1)); //点积，对k列非刚性系数，分别求每幅图像的平均值
    }
    //store connectivity
    if (con.size() > 0) { //default connectivity
        int m = con.size();
        C.create(m, 2, CV_32F);
        for (int i = 0; i < m; i++) {
            C.at<int>(i, 0) = con[i][0];
            C.at<int>(i, 1) = con[i][1];
        }
    } else {              //user-specified connectivity
        C.create(n, 2, CV_32S);
        for (int i = 0; i < n - 1; i++) {
            C.at<int>(i, 0) = i;
            C.at<int>(i, 1) = i + 1;
        }
        C.at<int>(n - 1, 0) = n - 1;
        C.at<int>(n - 1, 1) = 0;
    }
}

cv::Mat shap_model::procrustes(const cv::Mat &x, const int itol, const float ftol) {
    /* X.cols:特征点个数，X.rows: 图像数量*2 */
    int N = x.cols, n = x.rows / 2;
    cv::Mat Co, p = x.clone();                //copy
    for (int i = 0; i < N; ++i) {
        /*取X第i个列向量*/
        cv::Mat p = p.col(i);                 //i`th shape
        float mx = 0, my = 0;                //compute centre of mass...
        for (int j = 0; j < n; ++j) {        //for x and   y  separately
            mx += p.at(2 * j);
            my += p.at(2 * j + 1);
        }
        /*分别求图像集，2维空间坐标x和y的平均值*/
        mx /= n;
        my /= n;
        /*对x,y坐标去中心化*/
        for (int j = 0; j < n; ++j) {       //remov center of mass
            p.at(2 * j) -= mx;
            p.at(2 * j + 1) -= my;
        }
    }
    for (int iter = 0; iter < itol; ++iter) {
        /*计算（含n个形状）的重心*/
        cv::Mat C = p * cv::Mat::ones(N, 1, CV_32F) / N; //compute normalized..
        /*C为2n*1维矩阵，含n个重心，对n个重心归一化处理*/
        cv::normalize(C, C);                       //canonical  shape
        if (iter > 0) {
            //norm:求绝对范数，小于阈值，则退出循环
            if (norm(C, Co) < ftol) break;          //converged?
        }
        Co = C.clone();                             //remember current estimate
        for (int i = 0; i < N; ++i) {
            //求当前形状与归一化重心之间的旋转角度，即上式a和b
            cv::Mat R = this->rot_scale_align(p.col(i), C);
            for (int j = 0; j < n; ++j) {        //apply  similarity transform
                float x = p.at(2 * j, i), y = p.at(2 * j + 1, i);
                /*仿射变化*/
                p.at(2 * j, i) = R.at(0, 0) * x + R.at(0, 1) * y;
                p.at(2 * j + 1, i) = R.at(1, 0) * x + R.at(1, 1) * y;
            }
        }
    }

    return p;   //returned procrustes aligned shape
}

cv::Mat shap_model::calc_rigid_basis(const cv::Mat &X) {
    //compute mean shape
    int N = X.cols, n = X.rows / 2;
    cv::Mat mean = X * cv::Mat::ones(N, 1, CV_32F) / N;
    //construct basis for similarity transform
    cv::Mat R(2 * n, 4, CV_32F);
    for (int i = 0; i < n; i++) {
        R.at(2 * i, 0) = mean.at(2 * i);
        R.at(2 * i + 1, 0) = mean.at(2 * i + 1);
        R.at(2 * i, 1) = -mean.at(2 * i + 1);
        R.at(2 * i + 1, 1) = mean.at(2 * i);
        R.at(2 * i, 2) = 1.0;
        R.at(2 * i + 1, 2) = 0.0;
        R.at(2 * i, 3) = 0.0;
        R.at(2 * i + 1, 3) = 1.0;
    }
    //Gram-Schmidt orthonormalization
    for (int i = 0; i < 4; i++) {
        cv::Mat r = R.col(i);
        for (int j = 0; j < i; j++) {
            cv::Mat b = R.col(j);
            r -= b * (b.t() * r);
        }
        cv::normalize(r, r);
    }
    return R;
}

cv::Mat shap_model::pts2mat(const std::vector<std::vector<cv::Point2f> > &points) {

    int N = (int) points.size();
    /*检查图像数量大于0*/
    assert(N > 0);          //assert(expression)，如果括号内的表达式值为假，则打印出错信息
    int n = (int) points[0].size();
    for (int i = 1; i < N; i++)
        /*检查每幅图像的样本点数量是否相同*/
        assert(int(points[i].size()) == n);
    cv::Mat X(2 * n, N, CV_32F);
    for (int i = 0; i < N; i++) {
        cv::Mat x = X.col(i), y =
                cv::Mat(points[i]).reshape(1, 2 * n);  //points[i]是一个point2f向量，存放着一幅图像的样本点
        y.copyTo(x);
    }
    return X;
}

cv::Mat shap_model::rot_scale_align(const cv::Mat &src, const cv::Mat &dst) {
    //construct linear system
    int n = src.rows / 2;
    float a = 0, b = 0, d = 0;
    for (int i = 0; i < n; ++i) {
        d += src.at(2 * i) * src.at(2 * i) + src.at(2 * i + 1) * src.at(2 * i + 1);
        a += src.at(2 * i) * dst.at(2 * i) + src.at(2 * i + 1) * dst.at(2 * i + 1);
        b += src.at(2 * i) * dst.at(2 * i + 1) - src.at(2 * i + 1) * dst.at(2 * i);
    }
    a /= d;
    b /= d;      //solve linear system
    cv::Mat_<float> dd;
    dd = (cv::Mat_<float>(2, 2) << a, -b, b, a);
    return dd;
}

void shap_model::clamp(const float c) {
    /*p = V.t()*s;  simple projection
    * p: (k+4)*1维，存放采样点投影到联合分布空间的坐标集合
    */
    double scale = p.at(0);
    for (int i = 0; i < e.rows; i++) {
        if (e.at(i) < 0)
            continue;
        float v = c * sqrt(e.at(i));
        if (fabs(p.at(i) / scale) > v) {
            if (p.at(i) > 0)
                p.at(i) = v * scale;
            else
                p.at(i) = -v * scale;
        }
    }
}










