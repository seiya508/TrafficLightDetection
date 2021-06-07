package com.example.android.camera.utils.com.example.trafficlightdetection

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.RectF
import android.media.Image
import android.util.Log
import android.util.Size
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.trafficlightdetection.DetectionObject
import com.example.trafficlightdetection.ObjectDetector
import com.example.trafficlightdetection.OverlaySurfaceView
import com.example.trafficlightdetection.RoiCalculator
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op

/**
 * Analyze内の画像解析ユースケース
 * @param yuvToRgbConverter カメラ画像のImageバッファYUV_420_888からRGB形式に変換する
 * @param interpreter tfliteモデルを操作するライブラリ
 * @param labels 正解ラベルのリスト
 *
 * @param overlaySurfaceView Surface Viewのコールバック
 * @param imageProxySize 取得画像のサイズ
 * @param resultViewSize プレビューのサイズ
 */

class Analyze(
    private val yuvToRgbConverter: YuvToRgbConverter,
    private val interpreter: Interpreter,
    private val labels: List<String>,

    private var overlaySurfaceView: OverlaySurfaceView,
    private val resultViewSize: Size
) : ImageAnalysis.Analyzer {

    private var imageRotationDegrees: Int = 0

    @SuppressLint("UnsafeExperimentalUsageError")
    override fun analyze(imageProxy: ImageProxy) {
        if (imageProxy.image == null) return


        // TODO : 交差点接近

        // TODO : 車速の取得

        //取得画像の回転向き、大きさを取得
        imageRotationDegrees = imageProxy.imageInfo.rotationDegrees
        val imageProxySize = Size(imageProxy.width, imageProxy.height)

        // 物体検知器の作成
        val objectDetector = ObjectDetector(interpreter, labels, imageRotationDegrees)

        // TODO : ROI自動化
        val roi = RoiCalculator().calcRoi(imageProxySize)

        // 解析対象の画像を取得 (YUV -> RGB bitmap -> ROIで切り取る)
        val roiBitmap = cropBitmap(roi, yuvToRgbBitmap(imageProxy.image!!))

        imageProxy.close() // imageProxyの解放 : 必ず呼ぶ

        // 信号機検知処理(推論処理)
        // *detectedObjectListのバウンディングボックスはimageProxy座標になっている
        // 確率の高い順に格納されている
        val detectedObjectList = objectDetector.detect(roi, roiBitmap)

        // TODO : 信号機画像抜き出し (detectedObjectList, roiBitmap -> trafficLightBitmap)
        val trafficLightBitmap = roiBitmap

        // 色判定処理
        val redIsLighting = true //objectDetector.analyzeTrafficColor(trafficLightBitmap)

        // TODO : 警告通知処理 (, speed)

        // 検出結果の表示(OverlaySurfaceView.kt参照)
        overlaySurfaceView.draw(roi, detectedObjectList, redIsLighting, imageProxySize, resultViewSize)
    }


    // 画像をYUV -> RGB bitmap に変換する
    fun yuvToRgbBitmap(targetImage: Image): Bitmap {

        // YUVの生成
        val targetBitmap =
            Bitmap.createBitmap(targetImage.width, targetImage.height, Bitmap.Config.ARGB_8888)

        // RGB bitmapに変換
        yuvToRgbConverter.yuvToRgb(targetImage, targetBitmap)

        return targetBitmap
    }

    // ROIで切り取る
    fun cropBitmap(roi: RectF, targetBitmap: Bitmap): Bitmap{

        // ROIの領域を切り取る(ImageProxy座標)
        val roiBitmap = Bitmap.createBitmap(
            targetBitmap,
            roi.left.toInt(),
            roi.top.toInt(),
            (roi.right - roi.left).toInt(),
            (roi.bottom - roi.top).toInt(),
            null,
            true
        )

        Log.d("デバッグ", "targetBitmap.width:" + targetBitmap.width )
        Log.d("デバッグ", "targetBitmap.height:" + targetBitmap.height )

        Log.d("デバッグ", "roiBitmap.width:" + roiBitmap.width )
        Log.d("デバッグ", "roiBitmap.height:" + roiBitmap.height )

        return roiBitmap
    }
}