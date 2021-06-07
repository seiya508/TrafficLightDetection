package com.example.trafficlightdetection

import android.graphics.RectF
import android.util.Size

class RoiCalculator() {

    // TODO : ROI自動化
    // ROIの計算
    fun calcRoi(imageProxySize: Size): RectF {

        return RectF(
            // (ImageProxy座標)
            imageProxySize.width / 5f * 2,
            imageProxySize.height / 5f * 2,
            imageProxySize.width / 5f * 3,
            imageProxySize.height / 5f * 3f
        )
    }

}