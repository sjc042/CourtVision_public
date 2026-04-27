package com.courtvision.spike.pipeline

import android.graphics.Bitmap
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.extension.ExtendWith
import tech.apter.junit.jupiter.robolectric.RobolectricExtension

@ExtendWith(RobolectricExtension::class)
class FrameProcessorPerFramePerfTest {

    @Test
    fun personFrame_emitsRowWithPoseTimingsAndPoseSkippedFalse() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val fakeLogger = FakePerFramePerfLogger()
        val processor = FrameProcessor(
            scope = this,
            consumerDispatcher = dispatcher,
            perFrameLogger = fakeLogger
        )
        val fakePose = PerFrameFakePoseInferenceEngine(cannedPoseResult())
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)

        try {
            setPrivateField(processor, "poseInterpreter", fakePose)
            processor.runLivePoseForTest(bitmap, boxes = listOf(personBox()))
            processor.forceCurrentInferenceModeForTest(InferenceMode.GPU)
            processor.emitPerFrameRowForTest(
                frameTotalMs = 33.0,
                yoloPreprocessMs = 4.0,
                yoloInferenceMs = 14.0,
                yoloNmsMs = 2.0,
                rotationDegrees = 180
            )

            assertEquals(1, fakeLogger.rows.size)
            val row = fakeLogger.rows.first()
            assertFalse(row.poseSkipped)
            assertTrue(row.poseCropMs >= 0.0)
            assertEquals(1.0, row.posePreprocessMs, 0.0001)
            assertEquals(2.0, row.poseInferenceMs, 0.0001)
            assertEquals(1.0, row.posePostprocessMs, 0.0001)
            assertEquals("GPU", row.gpuMode)
            assertEquals(180, row.rotationDegrees)
        } finally {
            bitmap.recycle()
            processor.shutdown()
        }
    }

    @Test
    fun noPersonFrame_emitsRowWithPoseSkippedTrueAndZeroPoseStages() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val fakeLogger = FakePerFramePerfLogger()
        val processor = FrameProcessor(
            scope = this,
            consumerDispatcher = dispatcher,
            perFrameLogger = fakeLogger
        )
        val fakePose = PerFrameFakePoseInferenceEngine(cannedPoseResult())
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)

        try {
            setPrivateField(processor, "poseInterpreter", fakePose)
            processor.runLivePoseForTest(bitmap, boxes = listOf(ballBox()))
            processor.emitPerFrameRowForTest(
                frameTotalMs = 22.0,
                yoloPreprocessMs = 3.0,
                yoloInferenceMs = 11.0,
                yoloNmsMs = 1.5
            )

            assertEquals(1, fakeLogger.rows.size)
            val row = fakeLogger.rows.first()
            assertTrue(row.poseSkipped)
            assertEquals(0.0, row.poseCropMs, 0.0)
            assertEquals(0.0, row.posePreprocessMs, 0.0)
            assertEquals(0.0, row.poseInferenceMs, 0.0)
            assertEquals(0.0, row.posePostprocessMs, 0.0)
        } finally {
            bitmap.recycle()
            processor.shutdown()
        }
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
}

private class FakePerFramePerfLogger : PerFramePerfLogger {
    val rows = mutableListOf<PerFramePerfRow>()

    override val filePath: String = "/tmp/perframe.csv"

    override fun append(row: PerFramePerfRow) {
        rows += row
    }

    override fun close() = Unit
}

private class PerFrameFakePoseInferenceEngine(
    private val result: PoseResult
) : PoseInferenceEngine {
    val inferCalls = AtomicInteger(0)

    override fun infer(cropBitmap: Bitmap): PoseResult {
        inferCalls.incrementAndGet()
        return result
    }

    override fun close() = Unit
}
