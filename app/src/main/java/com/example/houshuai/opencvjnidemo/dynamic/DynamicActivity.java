package com.example.houshuai.opencvjnidemo.dynamic;

import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.view.SurfaceView;

import com.example.houshuai.opencvjnidemo.R;
import com.example.houshuai.opencvjnidemo.dynamic.impl.CameraVersionOneModle;
import com.example.houshuai.opencvjnidemo.dynamic.intel.IFaceRecognitionListener;

/**
 * @author 候帅
 *         Created by 候帅 on 2017/1/12. 15:11
 */

public class DynamicActivity extends AppCompatActivity {


    private SurfaceView mPreSurface;                        //原始Surface
    private SurfaceView mResuSuface;                        //处理后的Surface

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.acticity_dymic_deal);
        mPreSurface = (SurfaceView) findViewById(R.id.sv_dymic_deal_predata);
        mResuSuface = (SurfaceView) findViewById(R.id.sv_dymic_deal_result);
        initLogical();
    }

    /**
     * Method is deal for event's Logical
     */
    private void initLogical() {
        /**
         * {@link CameraVersionOneModle}  The result of processing the camera callback
         * */
        CameraVersionOneModle mPreCamera = new CameraVersionOneModle(mPreSurface, null);

        mPreCamera.setIFaceRecognition(new IFaceRecognitionListener() {
            /**
             * 另一个视图使用
             * */
            @Override
            public void faceDetected(int[] data) {


            }
        });

    }


}
