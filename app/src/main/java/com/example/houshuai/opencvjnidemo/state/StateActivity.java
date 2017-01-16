package com.example.houshuai.opencvjnidemo.state;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import com.example.houshuai.opencvjnidemo.R;

/**
 * @author 候帅
 *         Created by 候帅 on 2017/1/12. 15:11
 */

public class StateActivity extends AppCompatActivity {

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.acticity_state_deal);
        Button mButton = (Button) findViewById(R.id.btn_show_image);
        final ImageView mImageView = (ImageView) findViewById(R.id.iv_show_image);
        final StateJni stateJni = new StateJni();
        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                int[] ints = stateJni.dealStateImage(BitmapFactory.decodeResource(getResources(), R.drawable.image), 0x1);
                Bitmap bitmap = ((BitmapDrawable) getResources().getDrawable(R.drawable.image)).getBitmap();
                Bitmap resultBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
                resultBitmap.setPixels(ints,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
                mImageView.setImageBitmap(resultBitmap);
            }
        });
    }
}
