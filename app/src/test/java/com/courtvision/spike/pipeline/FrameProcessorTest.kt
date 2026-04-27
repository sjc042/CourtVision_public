package com.courtvision.spike.pipeline

import android.graphics.Bitmap
import android.graphics.Rect
import android.media.Image
import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import androidx.camera.core.impl.TagBundle
import androidx.camera.core.impl.utils.ExifData
import java.io.File
import java.io.RandomAccessFile
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertSame
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.extension.ExtendWith
import tech.apter.junit.jupiter.robolectric.RobolectricExtension

@ExtendWith(RobolectricExtension::class)
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

    @Test
    fun trackerMaxMissFrames_isForwarded() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)

        try {
            processor.setTrackerMaxMissFrames(17)
            testScheduler.runCurrent()

            val trackerField = FrameProcessor::class.java.getDeclaredField("ballTracker")
            trackerField.isAccessible = true
            val tracker = trackerField.get(processor) as KalmanBallTracker

            assertEquals(17, tracker.maxMissFrames)
        } finally {
            processor.shutdown()
        }
    }

    @Test
    fun stats_includeTrackingSnapshotFields() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)

        try {
            val trackedBall = TrackedBall(
                centroidX = 0.42f,
                centroidY = 0.61f,
                velocityX = 0.13f,
                velocityY = -0.08f,
                isTracked = true,
                rawBox = null
            )

            val trackedField = FrameProcessor::class.java.getDeclaredField("latestTrackedBall")
            trackedField.isAccessible = true
            @Suppress("UNCHECKED_CAST")
            val trackedRef = trackedField.get(processor) as java.util.concurrent.atomic.AtomicReference<TrackedBall?>
            trackedRef.set(trackedBall)

            val missField = FrameProcessor::class.java.getDeclaredField("latestMissStreak")
            missField.isAccessible = true
            val missRef = missField.get(processor) as AtomicInteger
            missRef.set(4)

            processor.publishWindowStatsForTest()
            val stats = processor.stats.value

            assertTrue(stats.trackingActive)
            assertEquals(0.42, stats.trackCx ?: -1.0, 0.0001)
            assertEquals(0.61, stats.trackCy ?: -1.0, 0.0001)
            assertEquals(0.13, stats.trackVx ?: -1.0, 0.0001)
            assertEquals(-0.08, stats.trackVy ?: -1.0, 0.0001)
            assertEquals(4, stats.missStreak)
            assertNotNull(stats.trackCx)
            assertNotNull(stats.trackCy)
        } finally {
            processor.shutdown()
        }
    }

    @Test
    fun resetInterpreter_clearsLastImageTimestampNs() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)

        try {
            val lastImageTimestampField =
                FrameProcessor::class.java.getDeclaredField("lastImageTimestampNs")
            lastImageTimestampField.isAccessible = true
            lastImageTimestampField.setLong(processor, 123_456_789L)

            processor.resetInterpreter()
            testScheduler.runCurrent()

            val updated = lastImageTimestampField.getLong(processor)
            assertEquals(-1L, updated)
        } finally {
            processor.shutdown()
        }
    }

    @Test
    fun submitImage_dropsFrames_whenPoseValidationMarkedComplete() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)
        val image = FakeImageProxy()

        try {
            val completeField = FrameProcessor::class.java.getDeclaredField("poseValidationComplete")
            completeField.isAccessible = true
            val complete = completeField.get(processor) as AtomicBoolean
            complete.set(true)

            processor.submitImage(image)

            assertTrue(image.isClosed)
            assertEquals(0, processor.stats.value.queueDepth)
        } finally {
            processor.shutdown()
        }
    }

    @Test
    fun submitPoseValidationBatch_emptyList_keepsValidationInactive() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)

        try {
            processor.submitPoseValidationBatch(emptyList())

            val activeField = FrameProcessor::class.java.getDeclaredField("poseValidationActive")
            activeField.isAccessible = true
            val active = activeField.get(processor) as AtomicBoolean
            assertFalse(active.get())
        } finally {
            processor.shutdown()
        }
    }

    @Test
    fun resetInterpreter_clearsPoseValidationFlags() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)

        try {
            val activeField = FrameProcessor::class.java.getDeclaredField("poseValidationActive")
            activeField.isAccessible = true
            val active = activeField.get(processor) as AtomicBoolean
            active.set(true)

            val completeField = FrameProcessor::class.java.getDeclaredField("poseValidationComplete")
            completeField.isAccessible = true
            val complete = completeField.get(processor) as AtomicBoolean
            complete.set(true)

            processor.resetInterpreter()
            testScheduler.runCurrent()

            assertFalse(active.get())
            assertFalse(complete.get())
        } finally {
            processor.shutdown()
        }
    }

    @Test
    fun processImage_liveYoloWithPerson_runsPoseAndEmitsResult() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)
        val fakePose = FakePoseInferenceEngine(cannedPoseResult())
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)

        try {
            setPrivateField(processor, "poseInterpreter", fakePose)
            processor.runLivePoseForTest(bitmap, boxes = listOf(personBox()))

            val overlay = processor.poseResult.value
            assertNotNull(overlay)
            assertEquals(33, overlay?.poseResult?.imageLandmarks33?.size)
            assertTrue((overlay?.cropRectNormalized?.right ?: 0f) > (overlay?.cropRectNormalized?.left ?: 0f))
            assertTrue(fakePose.inferCalls.get() > 0)
        } finally {
            bitmap.recycle()
            processor.shutdown()
        }
    }

    @Test
    fun runLivePoseForTest_emitsLivePoseOverlayWithCropRect() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)
        val fakePose = FakePoseInferenceEngine(cannedPoseResult())
        val bitmap = Bitmap.createBitmap(400, 600, Bitmap.Config.ARGB_8888)

        try {
            setPrivateField(processor, "poseInterpreter", fakePose)
            processor.runLivePoseForTest(bitmap, boxes = listOf(personBox()))

            val overlay = processor.poseResult.value
            assertNotNull(overlay)
            assertTrue((overlay?.cropRectNormalized?.left ?: -1f) >= 0f)
            assertTrue((overlay?.cropRectNormalized?.top ?: -1f) >= 0f)
            assertTrue((overlay?.cropRectNormalized?.right ?: 2f) <= 1f)
            assertTrue((overlay?.cropRectNormalized?.bottom ?: 2f) <= 1f)
        } finally {
            bitmap.recycle()
            processor.shutdown()
        }
    }

    @Test
    fun processImage_rotation90_sharesRotatedBitmapWithPose() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)
        val fakePose = FakePoseInferenceEngine(cannedPoseResult())
        val sensorBitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)

        try {
            setPrivateField(processor, "poseInterpreter", fakePose)
            val rotatedBitmap = rotateBitmapForDisplay(sensorBitmap, 90)
            try {
                processor.runLivePoseForTest(rotatedBitmap, boxes = listOf(personBox()))
            } finally {
                if (rotatedBitmap !== sensorBitmap) rotatedBitmap.recycle()
            }

            val rotatedForAssertion = rotateBitmapForDisplay(sensorBitmap, 90)
            val rotatedWidth = rotatedForAssertion.width
            val rotatedHeight = rotatedForAssertion.height
            if (rotatedForAssertion !== sensorBitmap) {
                rotatedForAssertion.recycle()
            }

            assertEquals(PoseTensorContract.INPUT_SIZE, fakePose.lastInputWidth.get())
            assertEquals(PoseTensorContract.INPUT_SIZE, fakePose.lastInputHeight.get())
            assertEquals(480, rotatedWidth)
            assertEquals(640, rotatedHeight)
        } finally {
            sensorBitmap.recycle()
            processor.shutdown()
        }
    }

    @Test
    fun processImage_liveYoloNoPerson_skipsPoseAndLeavesPoseResultNull() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)
        val fakePose = FakePoseInferenceEngine(cannedPoseResult())
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)

        try {
            setPrivateField(processor, "poseInterpreter", fakePose)
            processor.runLivePoseForTest(bitmap, boxes = listOf(ballBox()))

            assertNull(processor.poseResult.value)
            assertEquals(0, fakePose.inferCalls.get())
        } finally {
            bitmap.recycle()
            processor.shutdown()
        }
    }

    @Test
    fun processImage_bitmapRecycledInOuterFinally_noCrash() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)
        val fakePose = FakePoseInferenceEngine(cannedPoseResult())
        val firstBitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        val secondBitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)

        try {
            setPrivateField(processor, "poseInterpreter", fakePose)
            processor.runLivePoseForTest(firstBitmap, boxes = listOf(personBox()))
            processor.runLivePoseForTest(secondBitmap, boxes = listOf(personBox()))
            assertEquals(2, fakePose.inferCalls.get())
        } finally {
            firstBitmap.recycle()
            secondBitmap.recycle()
            processor.shutdown()
        }
    }

    @Test
    fun setPoseGatingMode_FSM_GATED_currentlyFallsBackToEveryFrame() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)
        val fakePose = FakePoseInferenceEngine(cannedPoseResult())
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)

        try {
            setPrivateField(processor, "poseInterpreter", fakePose)
            processor.setPoseGatingMode(PoseGatingMode.FSM_GATED)
            processor.runLivePoseForTest(bitmap, boxes = listOf(personBox()))

            assertNotNull(processor.poseResult.value)
            assertTrue(fakePose.inferCalls.get() > 0)
        } finally {
            bitmap.recycle()
            processor.shutdown()
        }
    }

    @Test
    fun initializePoseInterpreter_whenYoloGpu_usesGpuDelegate() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val capturedUseGpu = AtomicReference<Boolean?>(null)
        val factory = { _: MappedByteBuffer, useGpu: Boolean ->
            capturedUseGpu.set(useGpu)
            FakePoseInferenceEngine(cannedPoseResult())
        }
        val processor = FrameProcessor(
            scope = this,
            modelBufferProvider = null,
            poseModelBufferProvider = { createMappedByteBuffer() },
            poseInterpreterFactory = factory,
            consumerDispatcher = dispatcher
        )

        try {
            processor.forceCurrentInferenceModeForTest(InferenceMode.GPU)
            processor.initializePoseInterpreterForTest()

            assertEquals(true, capturedUseGpu.get())
            assertTrue(processor.hasPoseInterpreterForTest())
        } finally {
            processor.shutdown()
        }
    }

    @Test
    fun switchInterpreterCpuToGpu_closesAndReinitializesPose() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val created = mutableListOf<FakePoseInferenceEngine>()
        val factory = { _: MappedByteBuffer, _: Boolean ->
            FakePoseInferenceEngine(cannedPoseResult()).also { created.add(it) }
        }
        val processor = FrameProcessor(
            scope = this,
            modelBufferProvider = null,
            poseModelBufferProvider = { createMappedByteBuffer() },
            poseInterpreterFactory = factory,
            consumerDispatcher = dispatcher
        )

        try {
            processor.forceCurrentInferenceModeForTest(InferenceMode.CPU)
            processor.initializePoseInterpreterForTest()
            val first = created.first()

            processor.switchInterpreterForTest(InferenceMode.GPU)
            processor.forceCurrentInferenceModeForTest(InferenceMode.GPU)
            processor.initializePoseInterpreterForTest()
            val second = created.last()

            assertFalse(first === second)
            assertTrue(first.closed.get())
            assertTrue(processor.hasPoseInterpreterForTest())
            assertSame(second, created.last())
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

    private fun personBox() = DetectionBox(
        classId = 2,
        label = "person",
        confidence = 0.93f,
        left = 0.25f,
        top = 0.10f,
        right = 0.75f,
        bottom = 0.90f
    )

    private fun ballBox() = DetectionBox(
        classId = 0,
        label = "ball",
        confidence = 0.84f,
        left = 0.40f,
        top = 0.40f,
        right = 0.50f,
        bottom = 0.50f
    )

    private fun cannedPoseResult(): PoseResult {
        val imageLandmarks = (0 until PoseTensorContract.LANDMARKS_TOTAL).map {
            PoseImageLandmark(
                xPx = it.toFloat(),
                yPx = it.toFloat(),
                zPx = 0f,
                visibilityLogit = 1f,
                presenceLogit = 1f,
                visibility = 0.9f,
                presence = 0.95f
            )
        }
        val worldLandmarks = (0 until PoseTensorContract.LANDMARKS_TOTAL).map {
            PoseWorldLandmark(x = it.toFloat(), y = it.toFloat(), z = 0f)
        }
        return PoseResult(
            imageLandmarks39 = imageLandmarks,
            worldLandmarks39 = worldLandmarks,
            imageLandmarks33 = imageLandmarks.take(PoseTensorContract.LANDMARKS_CANONICAL),
            worldLandmarks33 = worldLandmarks.take(PoseTensorContract.LANDMARKS_CANONICAL),
            posePresenceLogit = 1.0f,
            posePresence = 0.73f,
            latency = PoseStageLatency(
                preprocessMs = 1.0,
                inferenceMs = 2.0,
                postprocessMs = 1.0,
                totalMs = 4.0
            )
        )
    }

    private fun setPrivateField(instance: Any, fieldName: String, value: Any?) {
        val field = instance::class.java.getDeclaredField(fieldName)
        field.isAccessible = true
        field.set(instance, value)
    }

    private fun createMappedByteBuffer(): MappedByteBuffer {
        val file = File.createTempFile("pose_test", ".bin")
        file.deleteOnExit()
        RandomAccessFile(file, "rw").use { raf ->
            raf.setLength(1024)
            raf.channel.use { channel ->
                return channel.map(FileChannel.MapMode.READ_ONLY, 0, raf.length())
            }
        }
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

private class FakePoseInferenceEngine(
    private val result: PoseResult
) : PoseInferenceEngine {
    val inferCalls = AtomicInteger(0)
    val closed = AtomicBoolean(false)
    val lastInputWidth = AtomicInteger(0)
    val lastInputHeight = AtomicInteger(0)

    override fun infer(cropBitmap: Bitmap): PoseResult {
        inferCalls.incrementAndGet()
        lastInputWidth.set(cropBitmap.width)
        lastInputHeight.set(cropBitmap.height)
        return result
    }

    override fun close() {
        closed.set(true)
    }
}

private fun FrameProcessor.publishWindowStatsForTest() {
    val method = FrameProcessor::class.java.getDeclaredMethod("publishWindowStats")
    method.isAccessible = true
    method.invoke(this)
}
