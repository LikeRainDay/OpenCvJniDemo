package com.example.houshuai.opencvjnidemo.bitmap_state;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.ImageView;

import com.example.houshuai.opencvjnidemo.R;

/**
 * @author 候帅
 *         Created by 候帅 on 2017/1/12. 15:11
 */

public class BitmapStateActivity extends AppCompatActivity {
    private static final int IMAGE_GRAY;                  //转化为灰度图
    private static final int IMAGE_ORGINAL;               //原图展示
    private static final int IMAGE_GAUSSION_BLUR;         //高斯模糊
    private static final int IMAGE_NON_RIGID_FACE_TRACK;  //非刚性人脸追踪

    static {
        IMAGE_ORGINAL = 0x1;
        IMAGE_NON_RIGID_FACE_TRACK = 0x4;
        IMAGE_GAUSSION_BLUR = 0x3;
        IMAGE_GRAY = 0x2;
    }


    private BitmapStateJni mJni;
    private ImageView mPreImageView, mResImageView;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.acticity_state_deal);
        //初始化
        init();
        dealLogic();
    }

    private void init() {
        mPreImageView = (ImageView) findViewById(R.id.iv_int_deal_predata);
        mResImageView = (ImageView) findViewById(R.id.iv_int_deal_result);
        mJni = new BitmapStateJni();  //初始化为JNI的调用
        mPreImageView.setImageResource(R.drawable.image);
    }

    private void dealLogic() {
        //原图展示
        showOldImage();
        //非刚性人脸追踪
        showState_Non_rigid_Face_Tracking();
        //转化为灰度图
        showGrayImage();
        //高斯模糊
        showState_GUSSION();

    }

    /**
     * Used to display the image information through the gray processing of Bitmap
     * 用于展示通过灰度处理最终后的Bitmap图像信息
     */
    private void showGrayImage() {
        findViewById(R.id.btn_int_show_Grass_image).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                showResult(IMAGE_GRAY);
            }
        });
    }

    /**
     * Use to display the image information og bitmap after filtering by non-right face tracking
     * 用于展示通过非刚性人脸最终后的Bitmap图像信息
     */
    private void showState_Non_rigid_Face_Tracking() {
        findViewById(R.id.btn_int_show_FaceDetel_image).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                showResult(IMAGE_NON_RIGID_FACE_TRACK);
            }
        });
    }

    /**
     * Use to display the image information og bitmap after filtering by non-right face tracking
     * 用于展示通过非刚性人脸最终后的Bitmap图像信息
     */
    private void showState_GUSSION() {
        findViewById(R.id.btn_int_show_GAUSSION_image).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                showResult(IMAGE_GAUSSION_BLUR);
            }
        });
    }

    /**
     * Used to display the image information of the Bitmap after filtering by JNI
     * 用于展示通过JNI过滤后的Bitmap的图像信息
     */
    private void showOldImage() {
        findViewById(R.id.btn_int_show_JNI_image).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                showResult(IMAGE_ORGINAL);
            }
        });
    }

    /**
     * Use to display the image information of the bitmap by type selection
     * 通过类型选择展示图片信息
     *
     * @param type 当前的图片处理类型
     */
    private void showResult(int type) {
        int[] ints = mJni.dealStateImage(BitmapFactory.decodeResource(getResources(), R.drawable.image), type);
        Bitmap bitmap = ((BitmapDrawable) getResources().getDrawable(R.drawable.image)).getBitmap();
        Bitmap resultBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.RGB_565);
        resultBitmap.setPixels(ints, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        mResImageView.setImageBitmap(resultBitmap);
    }
}
