package com.courtvision.spike.pipeline

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

/**
 * On-device smoke tests that measure per-component latency in the frame pipeline.
 * Run with: ./gradlew connectedDebugAndroidTest --tests "*.PreprocessingLatencyTest"
 *
 * Results appear in Logcat under tag "LatencyTest".
 */
@RunWith(AndroidJUnit4::class)
class PreprocessingLatencyTest {

    companion object {
        private const val TAG = "LatencyTest"
        private const val WARMUP_ITERATIONS = 5
        private const val MEASURED_ITERATIONS = 50
        private const val MODEL_INPUT_SIZE = 640
    }

    @Test
    fun parseModelOutput_latency() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)

        val output = syntheticOutput(
            listOf(
                Prediction(0.5f, 0.5f, 0.3f, 0.15f, 32, 0.91f),
                Prediction(0.25f, 0.25f, 0.19f, 0.38f, 0, 0.87f),
                Prediction(0.78f, 0.78f, 0.13f, 0.13f, 0, 0.72f),
                Prediction(0.1f, 0.9f, 0.08f, 0.08f, 32, 0.65f),
                Prediction(0.6f, 0.3f, 0.2f, 0.4f, 0, 0.80f)
            )
        )

        try {
            // Warmup
            repeat(WARMUP_ITERATIONS) {
                processor.parseModelOutput(output, 0.40f, 0.50f)
            }

            // Measure
            val durationsMs = mutableListOf<Double>()
            repeat(MEASURED_ITERATIONS) {
                val startNs = System.nanoTime()
                processor.parseModelOutput(output, 0.40f, 0.50f)
                durationsMs.add((System.nanoTime() - startNs) / 1_000_000.0)
            }

            logResults("parseModelOutput", durationsMs)
            assertTrue("p50 should be < 50ms", percentile(durationsMs, 0.50) < 50.0)
        } finally {
            processor.shutdown()
        }
    }

    @Test
    fun supportLibPreprocessing_latency_640x480() {
        benchmarkSupportLibPreprocessing(640, 480)
    }

    @Test
    fun supportLibPreprocessing_latency_1280x720() {
        benchmarkSupportLibPreprocessing(1280, 720)
    }

    @Test
    fun supportLibPreprocessing_latency_1920x1080() {
        benchmarkSupportLibPreprocessing(1920, 1080)
    }

    private fun benchmarkSupportLibPreprocessing(width: Int, height: Int) {
        val bitmap = createTestBitmap(width, height)
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()

        try {
            // Warmup
            repeat(WARMUP_ITERATIONS) {
                val tensorImage = TensorImage.fromBitmap(bitmap)
                imageProcessor.process(tensorImage)
            }

            // Measure
            val durationsMs = mutableListOf<Double>()
            repeat(MEASURED_ITERATIONS) {
                val startNs = System.nanoTime()
                val tensorImage = TensorImage.fromBitmap(bitmap)
                imageProcessor.process(tensorImage)
                durationsMs.add((System.nanoTime() - startNs) / 1_000_000.0)
            }

            logResults("supportLibPreprocessing(${width}x${height})", durationsMs)
            assertTrue("p50 should be < 100ms", percentile(durationsMs, 0.50) < 100.0)
        } finally {
            bitmap.recycle()
        }
    }

    private fun createTestBitmap(width: Int, height: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        // Fill with a gradient pattern so pixel data is realistic (not all zeros)
        val pixels = IntArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val r = (x * 255 / width)
                val g = (y * 255 / height)
                val b = ((x + y) * 255 / (width + height))
                pixels[y * width + x] = Color.rgb(r, g, b)
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    private fun syntheticOutput(predictions: List<Prediction>): Array<Array<FloatArray>> {
        val output = Array(1) { Array(84) { FloatArray(8400) } }
        predictions.forEachIndexed { index, prediction ->
            output[0][0][index] = prediction.cx
            output[0][1][index] = prediction.cy
            output[0][2][index] = prediction.w
            output[0][3][index] = prediction.h
            output[0][4 + prediction.classId][index] = prediction.confidence
        }
        return output
    }

    private fun logResults(label: String, durationsMs: List<Double>) {
        val sorted = durationsMs.sorted()
        val avg = durationsMs.average()
        val p50 = percentile(durationsMs, 0.50)
        val p95 = percentile(durationsMs, 0.95)
        val min = sorted.first()
        val max = sorted.last()

        Log.i(TAG, "=== $label ===")
        Log.i(TAG, "  iterations: ${durationsMs.size}")
        Log.i(TAG, "  avg:  ${"%.3f".format(avg)} ms")
        Log.i(TAG, "  p50:  ${"%.3f".format(p50)} ms")
        Log.i(TAG, "  p95:  ${"%.3f".format(p95)} ms")
        Log.i(TAG, "  min:  ${"%.3f".format(min)} ms")
        Log.i(TAG, "  max:  ${"%.3f".format(max)} ms")
    }

    private fun percentile(values: List<Double>, p: Double): Double {
        val sorted = values.sorted()
        val index = ((sorted.size - 1) * p).toInt()
        return sorted[index]
    }

    private data class Prediction(
        val cx: Float,
        val cy: Float,
        val w: Float,
        val h: Float,
        val classId: Int,
        val confidence: Float
    )
}
