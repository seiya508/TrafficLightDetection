package com.example.trafficlightdetection

import android.graphics.RectF

/**
 * 検出結果を入れるクラス
 */
data class DetectionObject(
    val score: Float,
    val label: String,
    val boundingBox: RectF
)