package com.example.houshuai.opencvjnidemo.int_state;

import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;

import com.example.houshuai.opencvjnidemo.R;
import com.example.houshuai.opencvjnidemo.bitmap_state.BitmapStateJni;

/**
 * @author 候帅
 *         Created by 候帅 on 2017/1/17. 10:31
 */

public class IntActivity extends AppCompatActivity implements View.OnClickListener {
    private static final String TAG = "IntActivity";
    private ImageView mPreImageView, mResImageView;
    private BitmapStateJni mJni;
    private IntJni mIntJni;
    private static final int IMAGE_GRAY;                  //转化为灰度图
    private static final int IMAGE_ORGINAL;               //原图展示
    private static final int IMAGE_GAUSSION_BLUR;         //高斯模糊
    private static final int IMAGE_NON_RIGID_FACE_TRACK;  //非刚性人脸追踪

    static {
        IMAGE_GRAY = 2;
        IMAGE_ORGINAL = 1;
        IMAGE_GAUSSION_BLUR = 3;
        IMAGE_NON_RIGID_FACE_TRACK = 4;
    }

    private int mWidth;
    private int mHeight;
    private int[] mPix;
    private int[] mInits;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.acticity_int_deal);
        //初始化控件
        initWidget();
    }

    private void initWidget() {
        mPreImageView = (ImageView) findViewById(R.id.iv_int_deal_predata);
        mResImageView = (ImageView) findViewById(R.id.iv_int_deal_result);
        mJni = new BitmapStateJni();  //初始化为JNI的调用
        mPreImageView.setImageResource(R.drawable.image);
        //初始化JNI的操作
        mIntJni = new IntJni();
        findViewById(R.id.btn_int_show_FaceDetel_image).setOnClickListener(this);
        findViewById(R.id.btn_int_show_Grass_image).setOnClickListener(this);
        findViewById(R.id.btn_int_show_GAUSSION_image).setOnClickListener(this);
        findViewById(R.id.btn_int_show_JNI_image).setOnClickListener(this);


    }

    private void initLogic() {
        long performance = System.currentTimeMillis();
        //初始化图片的数据
        Bitmap bitmap = ((BitmapDrawable) getResources().getDrawable(R.drawable.image)).getBitmap();
        mWidth = bitmap.getWidth();
        mHeight = bitmap.getHeight();
        mPix = new int[mWidth * mHeight];
        bitmap.getPixels(mPix, 0, mWidth, 0, 0, mWidth, mHeight);
        long l = System.currentTimeMillis() - performance;
        Log.e(TAG, "当前的初始化图片像素数据耗用时间为：" + l + "ms");
    }


    @Override
    public void onClick(View view) {
        //初始化图片的数据逻辑
        initLogic();
        long performance = System.currentTimeMillis();
        if (null == mInits) {
            mInits = new int[mWidth * mHeight];
        } else {
            mInits = null;
            mInits = new int[mWidth * mHeight];
        }
        switch (view.getId()) {
            case R.id.btn_int_show_FaceDetel_image: //
                mInits = mIntJni.intDetect(mPix, mWidth, mHeight, IMAGE_NON_RIGID_FACE_TRACK);
                break;
            case R.id.btn_int_show_Grass_image:
                mInits = mIntJni.intDetect(mPix, mWidth, mHeight, IMAGE_GRAY);
                break;
            case R.id.btn_int_show_GAUSSION_image:
                mInits = mIntJni.intDetect(mPix, mWidth, mHeight, IMAGE_GAUSSION_BLUR);
                break;
            case R.id.btn_int_show_JNI_image:
                mInits = mIntJni.intDetect(mPix, mWidth, mHeight, IMAGE_ORGINAL);
                break;
        }
        //进行加工展示
        Bitmap bitmap = Bitmap.createBitmap(mWidth, mHeight, Bitmap.Config.RGB_565);
        bitmap.setPixels(mInits, 0, mWidth, 0, 0, mWidth, mHeight);
        mResImageView.setImageBitmap(bitmap);


        long l = System.currentTimeMillis() - performance;
        Log.e(TAG, "JNi处理图片用时：" + l + "ms");
    }
}
