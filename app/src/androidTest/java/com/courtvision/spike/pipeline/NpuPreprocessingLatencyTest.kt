package com.courtvision.spike.pipeline

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * On-device smoke tests that measure candidate NPU preprocessing paths.
 * Run with: ./gradlew :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.courtvision.spike.pipeline.NpuPreprocessingLatencyTest
 *
 * Results appear in Logcat under tag "LatencyTest".
 */
class NpuPreprocessingLatencyTest {

    companion object {
        private const val TAG = "LatencyTest"
        private const val WARMUP_ITERATIONS = 5
        private const val MEASURED_ITERATIONS = 50
    }

    @Test
    fun manualNchwInt8_latency_640x480() {
        benchmarkStrategy("manualNchwInt8(640x480)", 640, 480) { processor, bitmap ->
            processor.preprocessNchwInt8Manual(bitmap)
        }
    }

    @Test
    fun manualNchwInt8_latency_1920x1080() {
        benchmarkStrategy("manualNchwInt8(1920x1080)", 1920, 1080) { processor, bitmap ->
            processor.preprocessNchwInt8Manual(bitmap)
        }
    }

    @Test
    fun transposeQuant_latency_640x480() {
        benchmarkStrategy("transposeQuant(640x480)", 640, 480) { processor, bitmap ->
            processor.preprocessNchwInt8Transpose(bitmap)
        }
    }

    @Test
    fun transposeQuant_latency_1920x1080() {
        benchmarkStrategy("transposeQuant(1920x1080)", 1920, 1080) { processor, bitmap ->
            processor.preprocessNchwInt8Transpose(bitmap)
        }
    }

    private fun benchmarkStrategy(
        label: String,
        width: Int,
        height: Int,
        strategy: (FrameProcessor, Bitmap) -> Unit
    ) = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)
        val bitmap = createTestBitmap(width, height)

        try {
            repeat(WARMUP_ITERATIONS) {
                strategy(processor, bitmap)
            }

            val durationsMs = mutableListOf<Double>()
            repeat(MEASURED_ITERATIONS) {
                val startNs = System.nanoTime()
                strategy(processor, bitmap)
                durationsMs += (System.nanoTime() - startNs) / 1_000_000.0
            }

            logResults(label, durationsMs)
            assertTrue("p50 should be < 50ms", percentile(durationsMs, 0.50) < 50.0)
        } finally {
            bitmap.recycle()
            processor.shutdown()
        }
    }

    private fun createTestBitmap(width: Int, height: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val r = x * 255 / width
                val g = y * 255 / height
                val b = (x + y) * 255 / (width + height)
                pixels[y * width + x] = Color.rgb(r, g, b)
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
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
}
