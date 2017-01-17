package com.example.houshuai.opencvjnidemo.int_state;

/**
 * @author 候帅
 *         Created by 候帅 on 2017/1/17. 10:31
 */

public class IntJni {



    static {
        System.loadLibrary("int-deal");
        System.loadLibrary("opencv_java3");
    }

    public final native int[] intDetect(int[] data, int weight, int height,int type);


}
