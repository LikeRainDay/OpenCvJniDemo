package com.example.houshuai.opencvjnidemo.dynamic;

/**
 * @author 候帅
 *         Created by 候帅 on 2017/1/12. 15:10
 */

public class DynamicJni {

    static {
        System.loadLibrary("dynamic-deal");
        System.loadLibrary("opencv_java3");

    }

    /**
     * @param bytes  当前摄像头回调的数据
     * @param height 当前获取摄像头回调数据的高度
     * @param width  当前获取摄像头回调数据的宽度
     * @return 返回使用到的图像数据（进行人脸位置的数据点更新）
     */
    public final native int[] dynamicDetect(byte[] bytes, int width, int height);


}
