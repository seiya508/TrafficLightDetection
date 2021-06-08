package com.example.trafficlightdetection

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.Manifest
import android.app.Application
import android.content.pm.PackageManager
import android.content.res.AssetFileDescriptor
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.camera.camera2.Camera2Config
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import com.example.android.camera.utils.com.example.trafficlightdetection.Analyze
import com.example.android.camera.utils.com.example.trafficlightdetection.YuvToRgbConverter
import com.google.common.util.concurrent.ListenableFuture
import kotlinx.android.synthetic.main.activity_main.*
import org.opencv.android.OpenCVLoader
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService

class MainActivity : AppCompatActivity() {
    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

        // モデル名とラベル名
        private const val MODEL_FILE_NAME = "ssd_mobilenet_v1.tflite"
        private const val LABEL_FILE_NAME = "coco_dataset_labels.txt"

        // 取得画像解像度
        // システムによって変更されるため、ObjectDetector.kt内で取得画像から再取得
        private const val imageProxyWidth = 1920
        private const val imageProxyHeight = 1080
    }

    private lateinit var cameraExecutor: ExecutorService

    // Surface Viewのコールバックをセット
    private lateinit var overlaySurfaceView: OverlaySurfaceView

    // CameraProvider
    private lateinit var cameraProviderFuture : ListenableFuture<ProcessCameraProvider>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        Log.d("デバッグ", "Hello")

        if( OpenCVLoader.initDebug() ){
            Log.d("デバッグ", "Opencv Succeed")
        }

        Log.d("デバッグ", "Hello2")

        overlaySurfaceView = OverlaySurfaceView(resultView)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // CameraProvider をリクエストする
        cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview : Preview = Preview.Builder()
                .build()

//          // Select back camera as a default
//          val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            val cameraSelector : CameraSelector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build()

            preview.setSurfaceProvider(cameraView.createSurfaceProvider())

            // 画像解析(今回は物体検知)のユースケース
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetRotation(cameraView.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST) // 最新のcameraのプレビュー画像だけをを流す
                .setTargetResolution(Size(imageProxyWidth, imageProxyHeight))
                .build()
                // 推論処理へ移動 (ObjectDetector.kt参照)
                .also {
                    it.setAnalyzer(
                        cameraExecutor,

                        // 画像解析(ObjectDetector.kt参照)
                        Analyze(
                            yuvToRgbConverter,
                            interpreter,
                            labels,
                            overlaySurfaceView,
                            Size(resultView.width, resultView.height)
                        )

                    )
                }

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer)

            } catch(exc: Exception) {
                Log.e("CameraX", "Use case binding failed", exc)
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
    private fun loadLabels(fileName: String = LABEL_FILE_NAME): List<String> {
        var labels = listOf<String>()
        var inputStream: InputStream? = null
        try {
            inputStream = assets.open(fileName)
            val reader = BufferedReader(InputStreamReader(inputStream))
            labels = reader.readLines()
        } catch (e: Exception) {
            Toast.makeText(this, "モデルデータを読み込めませんでした", Toast.LENGTH_SHORT).show()
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