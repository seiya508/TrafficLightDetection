package com.example.trafficlightdetection

import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
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
 * @param imageRotationDegrees
 */

class ObjectDetector(
    private val interpreter: Interpreter,
    private val labels: List<String>,
    private var imageRotationDegrees: Int = 0
) {

    companion object {
        private const val TAG = "ObjectDetector"

        // モデルのinputとoutputサイズ
        private const val IMG_SIZE_X = 300
        private const val IMG_SIZE_Y = 300
        private const val MAX_DETECTION_NUM = 10

        // 今回使うtfliteモデルは量子化済みなのでnormalize関連は127.5fではなく以下の通り
        private const val NORMALIZE_MEAN = 0f
        private const val NORMALIZE_STD = 1f

        // ===== 適宜変更 =====
        // 検出対象
        private const val DETECTION_TARGET = "traffic light"
        // 検出結果のスコアしきい値
        private const val SCORE_THRESHOLD = 0.1f
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
    // 画像をRGB bitmap  -> tensorflowImage -> tensorflowBufferに変換して推論し結果をリストとして出力
    fun detect(roi: RectF, roiBitmap: Bitmap): List<DetectionObject> {

        tfImageBuffer.load(roiBitmap)
        val tensorImage = tfImageProcessor.process(tfImageBuffer)

        //tfliteモデルで推論の実行
        interpreter.runForMultipleInputsOutputs(arrayOf(tensorImage.buffer), outputMap)

        // 推論結果を整形してリストにして返す
        val detectedObjectList = arrayListOf<DetectionObject>()

        loop@ for (i in 0 until outputDetectionNum[0].toInt()) {
            val score = outputScores[0][i]
            val label = labels[outputLabels[0][i].toInt()]

            // バウンディングボックスの計算(roiBitmap座標)
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

                Log.d("Debug", "DOL.BB : [" + i + "] " + detectedObjectList[i].boundingBox)

            } else {
                // 検出結果はスコアの高い順にソートされたものが入っているので、しきい値を下回ったらループ終了
                break@loop
            }
        }

        return detectedObjectList.take(4)
    }


    // ===== 色判定処理 =====
    fun analyzeTrafficColor(inputImage: Bitmap): Boolean{

        // bitmap to mat(rgb)
        val inputMat = Mat()
        Utils.bitmapToMat(inputImage, inputMat)

        // convert rgb to hsv
        Imgproc.cvtColor(inputMat, inputMat, Imgproc.COLOR_RGB2HSV)
        val outputMat = Mat()

        // only show the red area
        Core.inRange(inputMat, Scalar(0.0, 100.0, 100.0), Scalar(10.0, 255.0, 255.0), outputMat)
        Log.d(TAG, "row: ${outputMat.rows()}, col: ${outputMat.cols()}")
        Log.d(TAG, "all number: ${outputMat.rows() * outputMat.cols()}")

        // remove noises
        //Imgproc.blur(outputMat, outputMat, Size(10.0, 10.0))

        // convert output to binary
        Imgproc.threshold(outputMat, outputMat, 80.0, 255.0, Imgproc.THRESH_BINARY)

        return voteTrafficColor(outputMat)
    }

    private fun voteTrafficColor(image: Mat): Boolean {
        val labelImg = Mat()
        val stats = Mat()
        val centroids = Mat()
        val labelNum = Imgproc.connectedComponentsWithStats(image, labelImg, stats, centroids)

        var vote = 0
        var allArea = 0
        for(i in 0..labelNum) {
            val result = IntArray(labelNum)
            stats.get(i, Imgproc.CC_STAT_AREA, result)
            Log.d(TAG, "$i, ${result[0]}")
            if(i == 0) allArea = result[0]
            else vote += result[0]
        }
        return vote * 100 > allArea
    }

}