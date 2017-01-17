package com.example.houshuai.opencvjnidemo;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;

import com.example.houshuai.opencvjnidemo.dynamic.DynamicActivity;
import com.example.houshuai.opencvjnidemo.bitmap_state.BitmapStateActivity;
import com.example.houshuai.opencvjnidemo.int_state.IntActivity;

public class MainActivity extends AppCompatActivity implements OnClickListener {


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button mDynamic = (Button) findViewById(R.id.btn_dynamic_button);
        Button mState = (Button) findViewById(R.id.btn_state_button);
        Button mInt = (Button) findViewById(R.id.btn_int_button);
        mDynamic.setOnClickListener(this);
        mState.setOnClickListener(this);
        mInt.setOnClickListener(this);
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.btn_dynamic_button:
                skipActivity(DynamicActivity.class);
                break;
            case R.id.btn_state_button:
                skipActivity(BitmapStateActivity.class);
                break;
            case R.id.btn_int_button:
                skipActivity(IntActivity.class);
                break;
        }
    }

    private void skipActivity(Class mClass) {
        Intent intent = new Intent(this, mClass);
        startActivity(intent);
    }

}
