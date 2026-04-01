package com.courtvision.spike.pipeline

import android.graphics.Rect
import android.media.Image
import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import androidx.camera.core.impl.TagBundle
import androidx.camera.core.impl.utils.ExifData
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FrameProcessorTest {

    @Test
    fun submitFrame_overflow_counts_once_per_evicted_frame() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)

        try {
            processor.submitFrame(frame(1))
            processor.submitFrame(frame(2))
            processor.submitFrame(frame(3))

            processor.publishWindowStatsForTest()

            assertEquals(2L, processor.stats.value.droppedFrames)
        } finally {
            processor.shutdown()
        }
    }

    @Test
    fun submitFrame_success_sets_queueDepth_to_one() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)

        try {
            processor.submitFrame(frame(1))

            processor.publishWindowStatsForTest()

            assertEquals(1, processor.stats.value.queueDepth)
        } finally {
            processor.shutdown()
        }
    }

    @Test
    fun parseModelOutput_extractsExpectedBoxes_fromSyntheticTensor() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)
        val output = syntheticOutput(
            predictions = listOf(
                // ball class (0), high confidence
                Prediction(
                    cx = 0.5f,
                    cy = 0.5f,
                    w = 0.3125f,
                    h = 0.15625f,
                    classId = 0,
                    confidence = 0.91f
                ),
                // person class (2), high confidence
                Prediction(
                    cx = 0.25f,
                    cy = 0.25f,
                    w = 0.1875f,
                    h = 0.375f,
                    classId = 2,
                    confidence = 0.87f
                ),
                // shoot class (4) — all classes pass through now
                Prediction(
                    cx = 0.78f,
                    cy = 0.78f,
                    w = 0.125f,
                    h = 0.125f,
                    classId = 4,
                    confidence = 0.95f
                )
            )
        )

        try {
            val boxes = processor.parseModelOutput(
                output = output,
                confidenceThreshold = 0.40f,
                iouThreshold = 0.50f
            )

            assertEquals(3, boxes.size)
            assertTrue(boxes.any { it.classId == 0 })
            assertTrue(boxes.any { it.classId == 2 })
            assertTrue(boxes.any { it.classId == 4 })

            val ball = boxes.first { it.classId == 0 }
            assertEquals("ball", ball.label)
            assertEquals(0.91f, ball.confidence, 0.0001f)
            assertEquals(0.34375f, ball.left, 0.0001f)
            assertEquals(0.421875f, ball.top, 0.0001f)
            assertEquals(0.65625f, ball.right, 0.0001f)
            assertEquals(0.578125f, ball.bottom, 0.0001f)
        } finally {
            processor.shutdown()
        }
    }

    @Test
    fun rotateCoordinates_0_degrees() {
        val input = baseBox()
        val rotated = rotateDetectionBox(input, 0)

        assertBoxEquals(expected = input, actual = rotated)
    }

    @Test
    fun rotateCoordinates_90_degrees() {
        val rotated = rotateDetectionBox(baseBox(), 90)

        assertEquals(0.5f, rotated.left, 0.0001f)
        assertEquals(0.2f, rotated.top, 0.0001f)
        assertEquals(0.9f, rotated.right, 0.0001f)
        assertEquals(0.8f, rotated.bottom, 0.0001f)
    }

    @Test
    fun rotateCoordinates_180_degrees() {
        val rotated = rotateDetectionBox(baseBox(), 180)

        assertEquals(0.2f, rotated.left, 0.0001f)
        assertEquals(0.5f, rotated.top, 0.0001f)
        assertEquals(0.8f, rotated.right, 0.0001f)
        assertEquals(0.9f, rotated.bottom, 0.0001f)
    }

    @Test
    fun rotateCoordinates_270_degrees() {
        val rotated = rotateDetectionBox(baseBox(), 270)

        assertEquals(0.1f, rotated.left, 0.0001f)
        assertEquals(0.2f, rotated.top, 0.0001f)
        assertEquals(0.5f, rotated.right, 0.0001f)
        assertEquals(0.8f, rotated.bottom, 0.0001f)
    }

    @Test
    fun submitImage_skipped_when_model_not_confirmed() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)
        val analyzer = SpikeImageAnalyzer(processor) { false }
        val image = FakeImageProxy()

        try {
            analyzer.analyze(image)
            processor.publishWindowStatsForTest()

            assertTrue(image.isClosed)
            assertEquals(0, processor.stats.value.queueDepth)
        } finally {
            processor.shutdown()
        }
    }

    @Test
    fun submitImage_processed_after_model_confirmed() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)
        val analyzer = SpikeImageAnalyzer(processor) { true }
        val image = FakeImageProxy()

        try {
            analyzer.analyze(image)
            processor.publishWindowStatsForTest()

            assertFalse(image.isClosed)
            assertEquals(1, processor.stats.value.queueDepth)
        } finally {
            processor.shutdown()
        }
    }

    private fun frame(id: Int) = FramePacket(
        timestampNs = id.toLong(),
        width = 1280,
        height = 720,
        rotationDegrees = 0,
        format = 35
    )

    private fun syntheticOutput(predictions: List<Prediction>): Array<Array<FloatArray>> {
        val output = Array(1) { Array(9) { FloatArray(8400) } }  // 4 coords + 5 classes
        predictions.forEachIndexed { index, prediction ->
            output[0][0][index] = prediction.cx
            output[0][1][index] = prediction.cy
            output[0][2][index] = prediction.w
            output[0][3][index] = prediction.h
            output[0][4 + prediction.classId][index] = prediction.confidence
        }
        return output
    }

    private fun baseBox() = DetectionBox(
        classId = 0,
        label = "ball",
        confidence = 0.9f,
        left = 0.2f,
        top = 0.1f,
        right = 0.8f,
        bottom = 0.5f
    )

    private fun assertBoxEquals(expected: DetectionBox, actual: DetectionBox) {
        assertEquals(expected.left, actual.left, 0.0001f)
        assertEquals(expected.top, actual.top, 0.0001f)
        assertEquals(expected.right, actual.right, 0.0001f)
        assertEquals(expected.bottom, actual.bottom, 0.0001f)
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

private class FakeImageProxy : ImageProxy {
    var isClosed: Boolean = false
        private set

    private val imageInfo = object : ImageInfo {
        override fun getTagBundle(): TagBundle {
            throw UnsupportedOperationException("Not used in this test")
        }

        override fun getTimestamp(): Long = 0L

        override fun getRotationDegrees(): Int = 0

        override fun populateExifData(exifBuilder: ExifData.Builder) = Unit
    }

    override fun close() {
        isClosed = true
    }

    override fun getCropRect(): Rect {
        throw UnsupportedOperationException("Not used in this test")
    }

    override fun setCropRect(rect: Rect?) = Unit

    override fun getFormat(): Int {
        throw UnsupportedOperationException("Not used in this test")
    }

    override fun getHeight(): Int {
        throw UnsupportedOperationException("Not used in this test")
    }

    override fun getWidth(): Int {
        throw UnsupportedOperationException("Not used in this test")
    }

    override fun getPlanes(): Array<ImageProxy.PlaneProxy> = emptyArray()

    override fun getImageInfo(): ImageInfo = imageInfo

    override fun getImage(): Image? = null
}

private fun FrameProcessor.publishWindowStatsForTest() {
    val method = FrameProcessor::class.java.getDeclaredMethod("publishWindowStats")
    method.isAccessible = true
    method.invoke(this)
}
