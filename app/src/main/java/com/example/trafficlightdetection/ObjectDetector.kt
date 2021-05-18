package com.example.android.camera.utils.com.example.trafficlightdetection

import android.R.attr.bitmap
import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.Rect
import android.graphics.RectF
import android.media.Image
import android.util.Log
import android.util.Size
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.trafficlightdetection.DetectionObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op


typealias ObjectDetectorCallback = (image: List<DetectionObject>) -> Unit
/**
 * CameraXの物体検知の画像解析ユースケース
 * @param yuvToRgbConverter カメラ画像のImageバッファYUV_420_888からRGB形式に変換する
 * @param interpreter tfliteモデルを操作するライブラリ
 * @param labels 正解ラベルのリスト
 * @param resultViewSize 結果を表示するsurfaceViewのサイズ
 * @param listener コールバックで解析結果のリストを受け取る
 */
class ObjectDetector(
    private val yuvToRgbConverter: YuvToRgbConverter,
    private val interpreter: Interpreter,
    private val labels: List<String>,
    private val resultViewSize: Size,

    // ROI領域の設定
    private val roi: RectF,

    private val listener: ObjectDetectorCallback
) : ImageAnalysis.Analyzer {

    companion object {
        // モデルのinputとoutputサイズ
        private const val IMG_SIZE_X = 300
        private const val IMG_SIZE_Y = 300
        private const val MAX_DETECTION_NUM = 10

        // 今回使うtfliteモデルは量子化済みなのでnormalize関連は127.5fではなく以下の通り
        private const val NORMALIZE_MEAN = 0f
        private const val NORMALIZE_STD = 1f

        // ===== 適宜変更 =====
        // 検出対象番号( traffic light )
        private const val DETECTION_TARGET = "traffic light"
        // 検出結果のスコアしきい値
        private const val SCORE_THRESHOLD = 0.3f
    }

    private var imageRotationDegrees: Int = 0

    @SuppressLint("UnsafeExperimentalUsageError")
    override fun analyze(image: ImageProxy) {
        if (image.image == null) return

        imageRotationDegrees = image.imageInfo.rotationDegrees
        val detectedObjectList = detect(image.image!!)

        // ===== ここで推論処理(detect関数でコールバック) =====
        listener(detectedObjectList) //コールバックで検出結果を受け取る
        image.close()  //必ず呼ぶ：システムリソースの開放
    }

    private val tfImageProcessor by lazy {

        ImageProcessor.Builder()
            .add(ResizeOp(IMG_SIZE_X, IMG_SIZE_Y, ResizeOp.ResizeMethod.BILINEAR)) // モデルのinputに合うように画像のリサイズ
            .add(Rot90Op(-imageRotationDegrees / 90)) // 流れてくるImageProxyは90度回転しているのでその補正
            .add(NormalizeOp(NORMALIZE_MEAN, NORMALIZE_STD)) // normalization関連
            .build()
    }

    private val tfImageBuffer = TensorImage(DataType.UINT8)

    // 検出結果のバウンディングボックス [1:10:4]
    // バウンディングボックスは [top, left, bottom, right] の形
    private val outputBoundingBoxes: Array<Array<FloatArray>> = arrayOf(
        Array(MAX_DETECTION_NUM) {
            FloatArray(4)
        }
    )

    // 検出結果のクラスラベルインデックス [1:10]
    private val outputLabels: Array<FloatArray> = arrayOf(
        FloatArray(MAX_DETECTION_NUM)
    )

    // 検出結果の各スコア [1:10]
    private val outputScores: Array<FloatArray> = arrayOf(
        FloatArray(MAX_DETECTION_NUM)
    )

    // 検出した物体の数(今回はtflite変換時に設定されているので 10 (一定))
    private val outputDetectionNum: FloatArray = FloatArray(1)

    // 検出結果を受け取るためにmapにまとめる
    private val outputMap = mapOf(
        0 to outputBoundingBoxes,
        1 to outputLabels,
        2 to outputScores,
        3 to outputDetectionNum
    )

    // ===== 推論処理 =====
    // 画像をYUV -> RGB bitmap -> tensorflowImage -> tensorflowBufferに変換して推論し結果をリストとして出力
    private fun detect(targetImage: Image): List<DetectionObject> {
        var targetBitmap = Bitmap.createBitmap(targetImage.width, targetImage.height, Bitmap.Config.ARGB_8888)
        yuvToRgbConverter.yuvToRgb(targetImage, targetBitmap) // rgbに変換

        var croppedBitmap = targetBitmap

        Log.d("デバッグ", "roi:" + roi.left + " ?= " + roi.right )

        // ROIをクロップする(ResultView座標->ImageProxy座標に変換)
        val ipRoi = RectF(
            roi.left * 1600f/1080f,
            roi.top * 1200f/1536f,
            roi.right * 1600f/1080f,
            roi.bottom * 1200f/1536f
        // ImageProxy / ResultView
        )

        Log.d("デバッグ", "ipRoi:" + ipRoi.left + " ?= " + ipRoi.right )

        if( roi.left != roi.right ){
            croppedBitmap = Bitmap.createBitmap(
                targetBitmap,
                ipRoi.left.toInt(),
                ipRoi.top.toInt(),
                (ipRoi.right - ipRoi.left).toInt(),
                (ipRoi.bottom - ipRoi.top).toInt(),
                null,
                true
            )

            Log.d("デバッグ", "===== Cropped =====")
        }

        Log.d("デバッグ", "targetBitmap.width:" + targetBitmap.width )
        Log.d("デバッグ", "targetBitmap.height:" + targetBitmap.height )

        Log.d("デバッグ", "croppedBitmap.width:" + croppedBitmap.width )
        Log.d("デバッグ", "croppedBitmap.height:" + croppedBitmap.height )

        Log.d("デバッグ", "resultView.width:" + resultViewSize.width )
        Log.d("デバッグ", "resultView.height:" + resultViewSize.height )

        tfImageBuffer.load(croppedBitmap)
        val tensorImage = tfImageProcessor.process(tfImageBuffer)

        //tfliteモデルで推論の実行
        interpreter.runForMultipleInputsOutputs(arrayOf(tensorImage.buffer), outputMap)

        // ImageProxy座標->ResultViewへの座標変換
        var tmpX = resultViewSize.width.toFloat() / targetBitmap.width
        var tmpY = resultViewSize.height.toFloat() / targetBitmap.height

        // 推論結果を整形してリストにして返す
        val detectedObjectList = arrayListOf<DetectionObject>()
        loop@ for (i in 0 until outputDetectionNum[0].toInt()) {
            val score = outputScores[0][i]
            val label = labels[outputLabels[0][i].toInt()]

            var boundingBox = RectF(
                outputBoundingBoxes[0][i][1] * resultViewSize.width,
                outputBoundingBoxes[0][i][0] * resultViewSize.height,
                outputBoundingBoxes[0][i][3] * resultViewSize.width,
                outputBoundingBoxes[0][i][2] * resultViewSize.height
            )

            if( roi.left != roi.right ){
                boundingBox = RectF(
                    roi.left + outputBoundingBoxes[0][i][1] * IMG_SIZE_X * tmpX,
                    roi.top + outputBoundingBoxes[0][i][0] * IMG_SIZE_Y * tmpY,
                    roi.left + outputBoundingBoxes[0][i][3] * IMG_SIZE_X * tmpX,
                    roi.top + outputBoundingBoxes[0][i][2] * IMG_SIZE_Y * tmpY
                )
            }

            Log.d("デバッグ", "tmpX:" + tmpX )
            Log.d("デバッグ", "tmpY:" + tmpY )

            Log.d("デバッグ", "outputBoundingBoxesLeft:" + outputBoundingBoxes[0][i][1]
                    + " : " + outputBoundingBoxes[0][i][1] * resultViewSize.width)
            Log.d("デバッグ", "outputBoundingBoxesTop:" + outputBoundingBoxes[0][i][0]
                    + " : " + outputBoundingBoxes[0][i][0] * resultViewSize.height)
            Log.d("デバッグ", "outputBoundingBoxesRight:" + outputBoundingBoxes[0][i][3]
                    + " : " + outputBoundingBoxes[0][i][3] * resultViewSize.width)
            Log.d("デバッグ", "outputBoundingBoxesBottom:" + outputBoundingBoxes[0][i][2]
                    + " : " + outputBoundingBoxes[0][i][2] * resultViewSize.height)

            // 検出対象 かつ　しきい値よりも大きいもののみ追加
            if ( label == DETECTION_TARGET && score >= ObjectDetector.SCORE_THRESHOLD) {
                detectedObjectList.add(
                    DetectionObject(
                        score = score,
                        label = label,
                        boundingBox = boundingBox
                    )
                )
            } else {
                // 検出結果はスコアの高い順にソートされたものが入っているので、しきい値を下回ったらループ終了
                break@loop
            }
        }
        return detectedObjectList.take(4)
    }
}