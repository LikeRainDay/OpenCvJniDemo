package com.example.houshuai.opencvjnidemo.dynamic.impl;

import android.graphics.PixelFormat;
import android.hardware.Camera;
import android.os.SystemClock;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import com.example.houshuai.opencvjnidemo.dynamic.intel.IFaceRecognitionListener;


/**
 * @author 候帅
 *         Created by iyunwen on 2016/12/21.
 */

public class CameraVersionOneModle {
    private static final String TAG = CameraVersionOneModle.class.getSimpleName();


    private Camera mCamera;
    private IFaceRecognitionListener mFaceRecognition;
    private SurfaceView mSurfaceView;
    private Camera.Parameters parameters;
//    private final FaceRecognition mFaceDecete;
    private Thread mFaceThread;

    /**
     * @see #startRecoder  设置当前的是否开启识别
     */
    private boolean startRecoder;
    private final Object mObject = new Object();


    public void setIFaceRecognition(IFaceRecognitionListener mFaceRecognition) {
        this.mFaceRecognition = mFaceRecognition;
    }

    public CameraVersionOneModle(SurfaceView mSurfaceView, String filePath) {
        this.mSurfaceView = mSurfaceView;
        //创建对JNI的调用
//        new FaceRecognition();
//        mFaceDecete = new FaceRecognition(filePath);
        onCreate();
    }


    private void onCreate() {
        mSurfaceView.getHolder().setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
        // 设置surface的分辨率
        mSurfaceView.getHolder().setFixedSize(176, 144);
        // 设置屏幕常亮（必不可少）
        mSurfaceView.getHolder().setKeepScreenOn(true);

        mSurfaceView.getHolder().addCallback(new SurfaceCallBack());
    }

    private final class SurfaceCallBack implements SurfaceHolder.Callback {

        @Override
        public void surfaceCreated(SurfaceHolder surfaceHolder) {
            int cameraCount;
            Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
            cameraCount = Camera.getNumberOfCameras();
            //设置相机的参数
            for (int i = 0; i < cameraCount; i++) {
                Camera.getCameraInfo(i, cameraInfo);
                if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                    try {
                        mCamera = Camera.open(i);
                        mCamera.setPreviewDisplay(surfaceHolder);
//                        mContext.setCameraDisplayOrientation(i, mCamera);
                        //最重要的设置 帧图的回调
                        mCamera.setPreviewCallback(new MyPreviewCallback());
                        mCamera.startPreview();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        }


        private final class MyPreviewCallback implements Camera.PreviewCallback {
            @Override
            public void onPreviewFrame(final byte[] bytes, Camera camera) {
                synchronized (mObject) {
                    if (!startRecoder) {
                        startRecoder = true;
                        final Camera.Size previewSize = camera.getParameters().getPreviewSize();
                        //回调任连书
                        mFaceThread = new Thread(new Runnable() {
                            @Override
                            public void run() {







                                SystemClock.sleep(500);
                                startRecoder = false;
                            }
                        });
                        mFaceThread.start();
                    }
                }
            }
        }


        @Override
        public void surfaceChanged(SurfaceHolder surfaceHolder, int i, int width, int height) {
            if (mCamera != null) {
                parameters = mCamera.getParameters();
                parameters.setPictureFormat(PixelFormat.RGB_565);
                // 设置预览区域的大小
                parameters.setPreviewSize(width, height);
                // 设置每秒钟预览帧数
                parameters.setPreviewFpsRange(10, 20);
                // 设置预览图片的大小
                parameters.setPictureSize(width, height);
                parameters.setJpegQuality(80);
            }
        }

        @Override
        public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
            if (mCamera != null) {
                mCamera.setPreviewCallback(null);
                mCamera.stopPreview();
                mCamera.release();
                mCamera = null;
            }
            if (null != mFaceThread && mFaceThread.isAlive()) {
                mFaceThread.interrupt();
                mFaceThread = null;
            }
        }
    }

}

