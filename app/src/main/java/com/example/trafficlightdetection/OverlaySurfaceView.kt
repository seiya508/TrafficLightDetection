package com.example.trafficlightdetection

import android.graphics.*
import android.util.Size
import android.view.SurfaceHolder
import android.view.SurfaceView

/**
 * 検出結果を表示する透過surfaceView
 */
class OverlaySurfaceView(surfaceView: SurfaceView) :
    SurfaceView(surfaceView.context), SurfaceHolder.Callback {

    init {
        surfaceView.holder.addCallback(this)
        surfaceView.setZOrderOnTop(true)
    }

    private var surfaceHolder = surfaceView.holder
    private val paint = Paint()
    private val pathColorList = listOf(Color.RED, Color.GREEN, Color.CYAN, Color.BLUE)

    override fun surfaceCreated(holder: SurfaceHolder) {
        // surfaceViewを透過させる
        surfaceHolder.setFormat(PixelFormat.TRANSPARENT)
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
    }

    /**
     * surfaceViewに物体検出結果を表示
     */
    fun draw(
        roi: RectF,
        detectedObjectList: List<DetectionObject>,
        redIsLighting : Boolean,
        imageProxySize: Size,
        resultViewSize: Size
    ) {

        // surfaceHolder経由でキャンバス取得(画面がactiveでない時にもdrawされてしまいexception発生の可能性があるのでnullableにして以下扱ってます)
        val canvas: Canvas? = surfaceHolder.lockCanvas()
        // 前に描画していたものをクリア
        canvas?.drawColor(0, PorterDuff.Mode.CLEAR)

        // ImageProxy座標　-> ResultView座標への変換値
        val imageProxyToResultViewX = resultViewSize.width.toFloat() / imageProxySize.width
        val imageProxyToResultViewY = resultViewSize.height.toFloat() / imageProxySize.height

        // ImageProxy座標　-> ResultView座標への変換
        var ipRoi = Rect(
            (roi.left * imageProxyToResultViewX).toInt(),
            (roi.top * imageProxyToResultViewY).toInt(),
            (roi.right * imageProxyToResultViewX).toInt(),
            (roi.bottom * imageProxyToResultViewY).toInt()
        )

        // ROIを白い矩形で囲う
        paint.apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeWidth = 7f
            isAntiAlias = false
        }
        canvas?.drawRect(Rect(ipRoi.left, ipRoi.top, ipRoi.right, ipRoi.bottom), paint)

        // ROI以外を半透明にする
//        paint.apply {
//            color = Color.argb(127, 0, 0, 0)
//            style = Paint.Style.FILL
//            isAntiAlias = false
//        }
//        canvas?.drawRect(Rect(0, 0, ipRoi.right, ipRoi.top), paint)
//        canvas?.drawRect(Rect(0, ipRoi.top, ipRoi.left, resultViewSize.height), paint)
//        canvas?.drawRect(Rect(ipRoi.left, ipRoi.bottom, resultViewSize.width, resultViewSize.height), paint)
//        canvas?.drawRect(Rect(ipRoi.right, 0, resultViewSize.width, ipRoi.bottom), paint)

        detectedObjectList.mapIndexed { i, detectionObject ->

            // ImageProxy座標　-> ResultView座標への変換
            detectionObject.boundingBox = RectF(
                detectionObject.boundingBox.right * imageProxyToResultViewX,
                detectionObject.boundingBox.top * imageProxyToResultViewY,
                detectionObject.boundingBox.left * imageProxyToResultViewX,
                detectionObject.boundingBox.bottom * imageProxyToResultViewY
            )

            // バウンディングボックスの表示
            paint.apply {
                color = if(redIsLighting){ Color.RED }else{ Color.GREEN }
                style = Paint.Style.STROKE
                strokeWidth = 7f
                isAntiAlias = false
            }
            canvas?.drawRect(detectionObject.boundingBox, paint)

            // ラベルとスコアの表示
            paint.apply {
                style = Paint.Style.FILL
                isAntiAlias = true
                textSize = 77f
            }
            canvas?.drawText(
                detectionObject.label + " " + "%,.2f".format(detectionObject.score * 100) + "%",
                detectionObject.boundingBox.left,
                detectionObject.boundingBox.top - 5f,
                paint
            )
        }

        surfaceHolder.unlockCanvasAndPost(canvas ?: return)
    }
}