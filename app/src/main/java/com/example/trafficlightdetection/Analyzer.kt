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
import com.example.trafficlightdetection.OverlaySurfaceView
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op

/**
 * CameraXの物体検知の画像解析ユースケース
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
    override fun analyze(imageProxy: ImageProxy) {
        if (imageProxy.image == null) return

        imageRotationDegrees = imageProxy.imageInfo.rotationDegrees

        // TODO : 交差点接近

        // TODO : 車速の取得

        //取得画像の大きさを取得
        val imageProxySize = Size(imageProxy.width, imageProxy.height)

        // TODO : ROI自動化
        val roi = calcRoi(imageProxySize)

        // 解析対象の画像を取得 (YUV -> RGB bitmap -> ROIで切り取る)
        val roiBitmap = yuvToRoiBitmap(roi, imageProxy.image!!)
        imageProxy.close() // imageProxyの解放 : 必ず呼ぶ

        // 信号機検知処理(推論処理)
        // *detectedObjectListのバウンディングボックスはimageProxy座標になっている
        val detectedObjectList = detect(roi, roiBitmap)

        // TODO : 信号機画像抜き出し (detectedObjectList, roiBitmap -> trafficLightBitmap)

        // TODO : 色判定処理 (trafficLightBitmap -> lightingRed: boolean)

        // TODO : 警告通知処理 (lightingRed, speed)

        // 検出結果の表示(OverlaySurfaceView.kt参照)
        overlaySurfaceView.draw(roi, detectedObjectList, imageProxySize, resultViewSize)
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

    // TODO : ROI自動化
    // ROIの計算
    private fun calcRoi(imageProxySize: Size): RectF {

        return RectF(
            // (ImageProxy座標)
            imageProxySize.width / 5f * 2,
            imageProxySize.height / 5f * 2,
            imageProxySize.width / 5f * 3,
            imageProxySize.height / 5f * 3f
        )
    }

    // 画像をYUV -> RGB bitmap -> ROIで切り取る
    private fun yuvToRoiBitmap(roi: RectF, targetImage: Image): Bitmap{

        // YUVの生成
        val targetBitmap = Bitmap.createBitmap(targetImage.width, targetImage.height, Bitmap.Config.ARGB_8888)

        // RGB bitmapに変換
        yuvToRgbConverter.yuvToRgb(targetImage, targetBitmap)

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

    // ===== 推論処理 =====
    // 画像をRGB bitmap  -> tensorflowImage -> tensorflowBufferに変換して推論し結果をリストとして出力
    private fun detect(roi: RectF, roiBitmap: Bitmap): List<DetectionObject> {

        tfImageBuffer.load(roiBitmap)
        val tensorImage = tfImageProcessor.process(tfImageBuffer)

        //tfliteモデルで推論の実行
        interpreter.runForMultipleInputsOutputs(arrayOf(tensorImage.buffer), outputMap)

        // 推論結果を整形してリストにして返す
        val detectedObjectList = arrayListOf<DetectionObject>()
        loop@ for (i in 0 until outputDetectionNum[0].toInt()) {
            val score = outputScores[0][i]
            val label = labels[outputLabels[0][i].toInt()]

            // バウンディングボックスの計算 : 計算はroiとroiBitmapSizeで完結している
            val boundingBox = RectF(
                roi.left + outputBoundingBoxes[0][i][1] * roiBitmap.width,
                roi.top + outputBoundingBoxes[0][i][0] * roiBitmap.height,
                roi.left + outputBoundingBoxes[0][i][3] * roiBitmap.width,
                roi.top + outputBoundingBoxes[0][i][2] * roiBitmap.height
            )

            // 検出対象 かつ　しきい値よりも大きいもののみ追加
            if (label == DETECTION_TARGET && score >= SCORE_THRESHOLD) {
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