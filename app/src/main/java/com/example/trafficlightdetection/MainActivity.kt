package com.example.trafficlightdetection

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.Manifest
import android.content.pm.PackageManager
import android.content.res.AssetFileDescriptor
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import com.example.android.camera.utils.com.example.trafficlightdetection.ObjectDetector
import com.example.android.camera.utils.com.example.trafficlightdetection.YuvToRgbConverter
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import java.util.concurrent.ExecutorService

class MainActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "CameraXBasic"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

        // モデル名とラベル名
        private const val MODEL_FILE_NAME = "ssd_mobilenet_v1.tflite"
        private const val LABEL_FILE_NAME = "coco_dataset_labels.txt"
    }

    private lateinit var cameraExecutor: ExecutorService

    // Surface Viewのコールバックをセット
    private lateinit var overlaySurfaceView: OverlaySurfaceView

    //クロップ(ROI)の座標
    // left == right のときはROIを設定しない
    private var roi = RectF(
        (1600f/2f - 150f) * 1080f/1600f,
        (1200f/2f - 150f) * 1536f/1200f,
        (1600f/2f - 150f) * 1080f/1600f,
        (1200f/2f + 150f) * 1536f/1200f
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        Log.d("デバッグ", "roi:" + roi.left + " ?= " + roi.right )

        crop_button.setOnClickListener{
            if( roi.left == roi.right ){
                // ROIの設定
                roi.right = (1600f/2f + 150f) * 1080f/1600f
//                    RectF(
//                    // (ImageProxy座標)
//                    1600/2 - 150,
//                    1200/2 -150,
//                    1600/2 + 150,
//                    1200/2 + 150
//                    // (ResultView座標)
//                    (1600f/2f - 150f) * 1080f/1600f,
//                    (1200f/2f - 150f) * 1536f/1200f,
//                    (1600f/2f + 150f) * 1080f/1600f,
//                    (1200f/2f + 150f) * 1536f/1200f
//                )

            }else{
                roi.right = roi.left
            }
            Log.d("デバッグ", "Button pushed")
            Log.d("デバッグ", "roi:" + roi.left + " ?= " + roi.right )
        }

        overlaySurfaceView = OverlaySurfaceView(resultView)
        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(cameraView.createSurfaceProvider())
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            // 画像解析(今回は物体検知)のユースケース
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetRotation(cameraView.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST) // 最新のcameraのプレビュー画像だけをを流す
                .setTargetResolution(Size(1920, 1080))
                .build()
                // 推論処理へ移動 (ObjectDetector.kt参照)
                .also {
                    it.setAnalyzer(
                        cameraExecutor,
                        ObjectDetector(
                            yuvToRgbConverter,
                            interpreter,
                            labels,
                            Size(resultView.width, resultView.height),
                            roi
                        ) { detectedObjectList ->

                            // ===== 検出結果の表示(OverlaySurfaceView.kt参照) =====
                            overlaySurfaceView.draw(
                                detectedObjectList,
                                roi
                            )

                        }
                    )
                }

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray) {
        // リクエストコードの確認
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            // 許可されたらstartCamera()へ
            if (allPermissionsGranted()) {
                startCamera()
                // 許可されなかったらテキストを表示
            } else {
                Toast.makeText(this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    // ===== Tensorflow Lite で使うために追加 =====
    // tfliteモデルを扱うためのラッパーを含んだinterpreter
    private val interpreter: Interpreter by lazy {
        Interpreter(loadModel())
    }

    // モデルの正解ラベルリスト
    private val labels: List<String> by lazy {
        loadLabels()
    }

    // tfliteモデルをassetsから読み込む
    private fun loadModel(fileName: String = MODEL_FILE_NAME): ByteBuffer {
        lateinit var modelBuffer: ByteBuffer
        var file: AssetFileDescriptor? = null
        try {
            file = assets.openFd(fileName)
            val inputStream = FileInputStream(file.fileDescriptor)
            val fileChannel = inputStream.channel
            modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, file.startOffset, file.declaredLength)
        } catch (e: Exception) {
            Toast.makeText(this, "モデルファイル読み込みエラー", Toast.LENGTH_SHORT).show()
            finish()
        } finally {
            file?.close()
        }
        return modelBuffer
    }

    // モデルの正解ラベルデータをassetsから取得
    private fun loadLabels(fileName: String = MainActivity.LABEL_FILE_NAME): List<String> {
        var labels = listOf<String>()
        var inputStream: InputStream? = null
        try {
            inputStream = assets.open(fileName)
            val reader = BufferedReader(InputStreamReader(inputStream))
            labels = reader.readLines()
        } catch (e: Exception) {
            Toast.makeText(this, "txtファイル読み込みエラー", Toast.LENGTH_SHORT).show()
            finish()
        } finally {
            inputStream?.close()
        }
        return labels
    }

    // カメラのYUV画像をRGBに変換するコンバータ
    private val yuvToRgbConverter: YuvToRgbConverter by lazy {
        YuvToRgbConverter(this)
    }
}