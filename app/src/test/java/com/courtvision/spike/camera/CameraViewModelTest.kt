package com.courtvision.spike.camera

import android.app.Application
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelStore
import androidx.lifecycle.viewmodel.CreationExtras
import com.courtvision.spike.pipeline.DetectionFrame
import com.courtvision.spike.pipeline.FrameProcessorGateway
import com.courtvision.spike.pipeline.GpuProbeResult
import com.courtvision.spike.pipeline.GpuStatus
import com.courtvision.spike.pipeline.InferenceMode
import com.courtvision.spike.pipeline.LivePoseOverlay
import com.courtvision.spike.pipeline.PerformanceLogger
import com.courtvision.spike.pipeline.PipelineStats
import com.courtvision.spike.pipeline.PersonSelectionMode
import com.courtvision.spike.pipeline.PoseImageLandmark
import com.courtvision.spike.pipeline.PoseFrameResult
import com.courtvision.spike.pipeline.PoseGatingMode
import com.courtvision.spike.pipeline.PoseResult
import com.courtvision.spike.pipeline.PoseStageLatency
import com.courtvision.spike.pipeline.PoseWorldLandmark
import com.courtvision.spike.pipeline.PoseTensorContract
import com.courtvision.spike.pipeline.QnnProbeResult
import com.courtvision.spike.pipeline.QnnStatus
import com.courtvision.spike.pipeline.RotationTelemetry
import com.courtvision.spike.pipeline.CropRectNormalized
import com.courtvision.spike.pipeline.TrackingLogger
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.emptyFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.test.TestDispatcher
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.extension.AfterEachCallback
import org.junit.jupiter.api.extension.BeforeEachCallback
import org.junit.jupiter.api.extension.ExtensionContext
import org.junit.jupiter.api.extension.RegisterExtension

@OptIn(ExperimentalCoroutinesApi::class)
class CameraViewModelTest {

    @JvmField
    @RegisterExtension
    val mainDispatcherExtension = MainDispatcherExtension()

    @AfterEach
    fun tearDown() {
        CameraViewModel.testOverrides = null
    }

    @Test
    fun setModel_resetsInterpreterAndClearsStaleUiState() {
        val fakeProcessor = FakeFrameProcessor()
        CameraViewModel.testOverrides = CameraViewModel.TestOverrides(
            frameProcessor = fakeProcessor,
            performanceLogger = FakePerformanceLogger(),
            trackingLogger = FakeTrackingLogger(),
            modelPaths = listOf(MODEL_A, MODEL_B),
            initialModelPath = MODEL_A,
            gpuProbeResult = GpuProbeResult(status = GpuStatus.GPU_SUPPORTED),
            nnApiProbeResult = NNAPI_AVAILABLE
        )
        val handle = createViewModel()
        val viewModel = handle.viewModel

        try {
            viewModel.onCameraError("stale pipeline error")

            viewModel.setModel(MODEL_B)

            assertEquals(1, fakeProcessor.resetInterpreterCalls)
            assertEquals(MODEL_B, viewModel.uiState.value.selectedModel)
            assertTrue(viewModel.uiState.value.detectionFrame == DetectionFrame())
            assertNull(viewModel.uiState.value.lastError)
        } finally {
            handle.store.clear()
        }
    }

    @Test
    fun setModel_whenLocked_doesNotSwapOrResetInterpreter() {
        val fakeProcessor = FakeFrameProcessor()
        CameraViewModel.testOverrides = CameraViewModel.TestOverrides(
            frameProcessor = fakeProcessor,
            performanceLogger = FakePerformanceLogger(),
            trackingLogger = FakeTrackingLogger(),
            modelPaths = listOf(MODEL_A, MODEL_B),
            initialModelPath = MODEL_A,
            gpuProbeResult = GpuProbeResult(status = GpuStatus.GPU_SUPPORTED),
            nnApiProbeResult = NNAPI_AVAILABLE
        )
        val handle = createViewModel()
        val viewModel = handle.viewModel

        try {
            viewModel.confirmModel()

            viewModel.setModel(MODEL_B)

            assertEquals(0, fakeProcessor.resetInterpreterCalls)
            assertEquals(MODEL_A, viewModel.uiState.value.selectedModel)
        } finally {
            handle.store.clear()
        }
    }

    @Test
    fun init_exposesNnApiProbeResult_fromOverrides() {
        val fakeProcessor = FakeFrameProcessor()
        CameraViewModel.testOverrides = CameraViewModel.TestOverrides(
            frameProcessor = fakeProcessor,
            performanceLogger = FakePerformanceLogger(),
            trackingLogger = FakeTrackingLogger(),
            modelPaths = listOf(MODEL_A),
            initialModelPath = MODEL_A,
            gpuProbeResult = GpuProbeResult(status = GpuStatus.GPU_SUPPORTED),
            nnApiProbeResult = NNAPI_UNAVAILABLE
        )
        val handle = createViewModel()
        val viewModel = handle.viewModel

        try {
            assertEquals(NNAPI_UNAVAILABLE, viewModel.uiState.value.nnApiProbeResult)
            assertEquals(false, viewModel.uiState.value.nnApiAvailable)
        } finally {
            handle.store.clear()
        }
    }

    @Test
    fun setInferenceMode_nnApi_propagatesToProcessorAndUi() {
        val fakeProcessor = FakeFrameProcessor()
        CameraViewModel.testOverrides = CameraViewModel.TestOverrides(
            frameProcessor = fakeProcessor,
            performanceLogger = FakePerformanceLogger(),
            trackingLogger = FakeTrackingLogger(),
            modelPaths = listOf(MODEL_A),
            initialModelPath = MODEL_A,
            gpuProbeResult = GpuProbeResult(status = GpuStatus.GPU_SUPPORTED),
            nnApiProbeResult = NNAPI_AVAILABLE
        )
        val handle = createViewModel()
        val viewModel = handle.viewModel

        try {
            viewModel.setInferenceMode(InferenceMode.NNAPI)

            assertEquals(true, viewModel.uiState.value.nnApiAvailable)
            assertEquals(InferenceMode.NNAPI, fakeProcessor.lastSetMode)
            assertEquals(InferenceMode.NNAPI, viewModel.uiState.value.selectedMode)
        } finally {
            handle.store.clear()
        }
    }

    @Test
    fun uiState_poseOverlay_reflectsGatewayEmissions() = runTest {
        val fakeProcessor = FakeFrameProcessor()
        CameraViewModel.testOverrides = CameraViewModel.TestOverrides(
            frameProcessor = fakeProcessor,
            performanceLogger = FakePerformanceLogger(),
            trackingLogger = FakeTrackingLogger(),
            modelPaths = listOf(MODEL_A),
            initialModelPath = MODEL_A,
            gpuProbeResult = GpuProbeResult(status = GpuStatus.GPU_SUPPORTED),
            nnApiProbeResult = NNAPI_AVAILABLE
        )
        val handle = createViewModel()
        val viewModel = handle.viewModel

        try {
            val preConfirmOverlay = sampleOverlay()
            val postConfirmOverlay = sampleOverlay(
                cropRect = CropRectNormalized(
                    left = 0.15f,
                    top = 0.1f,
                    right = 0.85f,
                    bottom = 0.9f
                )
            )

            fakeProcessor.emitPoseOverlay(preConfirmOverlay)
            runCurrent()
            assertNull(viewModel.uiState.value.poseOverlay)

            viewModel.confirmModel()
            runCurrent()
            fakeProcessor.emitPoseOverlay(postConfirmOverlay)
            runCurrent()

            assertEquals(postConfirmOverlay, viewModel.uiState.value.poseOverlay)

            viewModel.restartSession()
            runCurrent()
            assertNull(viewModel.uiState.value.poseOverlay)
        } finally {
            handle.store.clear()
        }
    }

    @Test
    fun setTrackerMaxMissFrames_clampsAndPropagates() {
        val fakeProcessor = FakeFrameProcessor()
        CameraViewModel.testOverrides = CameraViewModel.TestOverrides(
            frameProcessor = fakeProcessor,
            performanceLogger = FakePerformanceLogger(),
            trackingLogger = FakeTrackingLogger(),
            modelPaths = listOf(MODEL_A),
            initialModelPath = MODEL_A,
            gpuProbeResult = GpuProbeResult(status = GpuStatus.GPU_SUPPORTED),
            nnApiProbeResult = NNAPI_AVAILABLE
        )
        val handle = createViewModel()
        val viewModel = handle.viewModel

        try {
            viewModel.setTrackerMaxMissFrames(99)

            assertEquals(30, fakeProcessor.lastSetMaxMissFrames)
            assertEquals(30, viewModel.uiState.value.trackerMaxMissFrames)
        } finally {
            handle.store.clear()
        }
    }

    @Test
    fun setTrackerNoise_propagatesToProcessorAndUi() {
        val fakeProcessor = FakeFrameProcessor()
        CameraViewModel.testOverrides = CameraViewModel.TestOverrides(
            frameProcessor = fakeProcessor,
            performanceLogger = FakePerformanceLogger(),
            trackingLogger = FakeTrackingLogger(),
            modelPaths = listOf(MODEL_A),
            initialModelPath = MODEL_A,
            gpuProbeResult = GpuProbeResult(status = GpuStatus.GPU_SUPPORTED),
            nnApiProbeResult = NNAPI_AVAILABLE
        )
        val handle = createViewModel()
        val viewModel = handle.viewModel

        try {
            viewModel.setTrackerNoise(1.23e-4f, 5.67e-2f)

            assertEquals(1.23e-4f, fakeProcessor.lastProcessNoise)
            assertEquals(5.67e-2f, fakeProcessor.lastMeasurementNoise)
            assertEquals(1.23e-4f, viewModel.uiState.value.trackerProcessNoise)
            assertEquals(5.67e-2f, viewModel.uiState.value.trackerMeasurementNoise)
        } finally {
            handle.store.clear()
        }
    }

    @Test
    fun onCleared_closesPerformanceAndTrackingLoggers() {
        val fakeProcessor = FakeFrameProcessor()
        val fakePerformanceLogger = FakePerformanceLogger()
        val fakeTrackingLogger = FakeTrackingLogger()
        CameraViewModel.testOverrides = CameraViewModel.TestOverrides(
            frameProcessor = fakeProcessor,
            performanceLogger = fakePerformanceLogger,
            trackingLogger = fakeTrackingLogger,
            modelPaths = listOf(MODEL_A),
            initialModelPath = MODEL_A,
            gpuProbeResult = GpuProbeResult(status = GpuStatus.GPU_SUPPORTED),
            nnApiProbeResult = NNAPI_AVAILABLE
        )
        val handle = createViewModel()

        handle.store.clear()

        assertTrue(fakePerformanceLogger.closed)
        assertTrue(fakeTrackingLogger.closed)
    }

    @Test
    fun startDay5PoseIsolatedValidation_emptySelection_setsFailureStatus() {
        val fakeProcessor = FakeFrameProcessor()
        CameraViewModel.testOverrides = CameraViewModel.TestOverrides(
            frameProcessor = fakeProcessor,
            performanceLogger = FakePerformanceLogger(),
            trackingLogger = FakeTrackingLogger(),
            modelPaths = listOf(MODEL_A),
            initialModelPath = MODEL_A,
            gpuProbeResult = GpuProbeResult(status = GpuStatus.GPU_SUPPORTED),
            nnApiProbeResult = NNAPI_AVAILABLE,
            poseOutputRootProvider = { File(System.getProperty("java.io.tmpdir"), "cv-pose-test") }
        )
        val handle = createViewModel()
        val viewModel = handle.viewModel

        try {
            viewModel.confirmModel()
            viewModel.startDay5PoseIsolatedValidation(emptyList())

            assertEquals(0, fakeProcessor.submitPoseValidationBatchCalls)
            assertEquals("FAILED: no images selected", viewModel.uiState.value.poseValidationStatus)
        } finally {
            handle.store.clear()
        }
    }

    @Test
    fun startDay5PoseIsolatedValidation_withoutModelConfirmation_setsFailureStatus() {
        val fakeProcessor = FakeFrameProcessor()
        CameraViewModel.testOverrides = CameraViewModel.TestOverrides(
            frameProcessor = fakeProcessor,
            performanceLogger = FakePerformanceLogger(),
            trackingLogger = FakeTrackingLogger(),
            modelPaths = listOf(MODEL_A),
            initialModelPath = MODEL_A,
            gpuProbeResult = GpuProbeResult(status = GpuStatus.GPU_SUPPORTED),
            nnApiProbeResult = NNAPI_AVAILABLE,
            poseOutputRootProvider = { File(System.getProperty("java.io.tmpdir"), "cv-pose-test") }
        )
        val handle = createViewModel()
        val viewModel = handle.viewModel

        try {
            viewModel.startDay5PoseIsolatedValidation(emptyList())

            assertEquals(0, fakeProcessor.submitPoseValidationBatchCalls)
            assertEquals(
                "FAILED: start inference first",
                viewModel.uiState.value.poseValidationStatus
            )
        } finally {
            handle.store.clear()
        }
    }

    @Test
    fun modelAssetPathForMode_routesQnnToInt8Asset_andOtherModesToSelectedModel() {
        val fakeProcessor = FakeFrameProcessor()
        CameraViewModel.testOverrides = CameraViewModel.TestOverrides(
            frameProcessor = fakeProcessor,
            performanceLogger = FakePerformanceLogger(),
            trackingLogger = FakeTrackingLogger(),
            modelPaths = listOf(MODEL_A),
            initialModelPath = MODEL_A,
            gpuProbeResult = GpuProbeResult(status = GpuStatus.GPU_SUPPORTED),
            nnApiProbeResult = NNAPI_AVAILABLE
        )
        val handle = createViewModel()
        val viewModel = handle.viewModel

        try {
            assertEquals(MODEL_A, viewModel.modelAssetPathForMode(InferenceMode.CPU))
            assertEquals(MODEL_A, viewModel.modelAssetPathForMode(InferenceMode.GPU))
            assertEquals(MODEL_A, viewModel.modelAssetPathForMode(InferenceMode.NNAPI))
            assertEquals(
                "spike_qai_yolo11n_640_5-class_04-28-2026_int8.tflite",
                viewModel.modelAssetPathForMode(InferenceMode.QNN_NPU)
            )
        } finally {
            handle.store.clear()
        }
    }

    @Test
    fun qnnProbeSupportedAndQuantized_setsQnnAvailableTrue() {
        val fakeProcessor = FakeFrameProcessor()
        CameraViewModel.testOverrides = CameraViewModel.TestOverrides(
            frameProcessor = fakeProcessor,
            performanceLogger = FakePerformanceLogger(),
            trackingLogger = FakeTrackingLogger(),
            modelPaths = listOf(MODEL_A),
            initialModelPath = MODEL_A,
            gpuProbeResult = GpuProbeResult(status = GpuStatus.GPU_SUPPORTED),
            qnnProbeResult = QnnProbeResult(
                status = QnnStatus.QNN_SUPPORTED,
                htpQuantizedSupported = true
            ),
            nnApiProbeResult = NNAPI_AVAILABLE
        )
        val handle = createViewModel()
        val viewModel = handle.viewModel

        try {
            assertTrue(viewModel.uiState.value.qnnAvailable)
            assertEquals(QnnStatus.QNN_SUPPORTED, viewModel.uiState.value.qnnProbeResult.status)
        } finally {
            handle.store.clear()
        }
    }

    @Test
    fun qnnProbeSupportedButFp16Only_setsQnnAvailableFalse() {
        val fakeProcessor = FakeFrameProcessor()
        CameraViewModel.testOverrides = CameraViewModel.TestOverrides(
            frameProcessor = fakeProcessor,
            performanceLogger = FakePerformanceLogger(),
            trackingLogger = FakeTrackingLogger(),
            modelPaths = listOf(MODEL_A),
            initialModelPath = MODEL_A,
            gpuProbeResult = GpuProbeResult(status = GpuStatus.GPU_SUPPORTED),
            qnnProbeResult = QnnProbeResult(
                status = QnnStatus.QNN_SUPPORTED,
                htpFp16Supported = true,
                htpQuantizedSupported = false
            ),
            nnApiProbeResult = NNAPI_AVAILABLE
        )
        val handle = createViewModel()
        val viewModel = handle.viewModel

        try {
            assertEquals(false, viewModel.uiState.value.qnnAvailable)
            assertEquals(true, viewModel.uiState.value.qnnProbeResult.htpFp16Supported)
            assertEquals(false, viewModel.uiState.value.qnnProbeResult.htpQuantizedSupported)
        } finally {
            handle.store.clear()
        }
    }

    @Test
    fun qnnProbeFailure_setsQnnAvailableFalse_andStoresProbeResult() {
        val fakeProcessor = FakeFrameProcessor()
        val probeResult = QnnProbeResult(
            status = QnnStatus.QNN_INIT_FAILED,
            reason = "probe failure"
        )
        CameraViewModel.testOverrides = CameraViewModel.TestOverrides(
            frameProcessor = fakeProcessor,
            performanceLogger = FakePerformanceLogger(),
            trackingLogger = FakeTrackingLogger(),
            modelPaths = listOf(MODEL_A),
            initialModelPath = MODEL_A,
            gpuProbeResult = GpuProbeResult(status = GpuStatus.GPU_SUPPORTED),
            qnnProbeResult = probeResult,
            nnApiProbeResult = NNAPI_AVAILABLE
        )
        val handle = createViewModel()
        val viewModel = handle.viewModel

        try {
            assertEquals(false, viewModel.uiState.value.qnnAvailable)
            assertEquals(probeResult, viewModel.uiState.value.qnnProbeResult)
        } finally {
            handle.store.clear()
        }
    }

    private fun createViewModel(): ViewModelHandle {
        val store = ViewModelStore()
        val factory = object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                return CameraViewModel(Application()) as T
            }

            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(
                modelClass: Class<T>,
                extras: CreationExtras
            ): T {
                return CameraViewModel(Application()) as T
            }
        }
        val viewModel = ViewModelProvider(store, factory)[CameraViewModel::class.java]
        return ViewModelHandle(store = store, viewModel = viewModel)
    }

    private fun sampleOverlay(
        cropRect: CropRectNormalized = CropRectNormalized(
            left = 0.1f,
            top = 0.1f,
            right = 0.8f,
            bottom = 0.9f
        )
    ): LivePoseOverlay {
        val imageLandmarks = (0 until PoseTensorContract.LANDMARKS_TOTAL).map { index ->
            PoseImageLandmark(
                xPx = index.toFloat(),
                yPx = index.toFloat(),
                zPx = 0f,
                visibilityLogit = 1f,
                presenceLogit = 1f,
                visibility = 0.9f,
                presence = 0.95f
            )
        }
        val worldLandmarks = (0 until PoseTensorContract.LANDMARKS_TOTAL).map { index ->
            PoseWorldLandmark(x = index.toFloat(), y = index.toFloat(), z = 0f)
        }
        val result = PoseResult(
            imageLandmarks39 = imageLandmarks,
            worldLandmarks39 = worldLandmarks,
            imageLandmarks33 = imageLandmarks.take(PoseTensorContract.LANDMARKS_CANONICAL),
            worldLandmarks33 = worldLandmarks.take(PoseTensorContract.LANDMARKS_CANONICAL),
            posePresenceLogit = 1f,
            posePresence = 0.73f,
            latency = PoseStageLatency(1.0, 2.0, 1.0, 4.0)
        )
        return LivePoseOverlay(
            poseResult = result,
            cropRectNormalized = cropRect
        )
    }

    private class FakePerformanceLogger : PerformanceLogger {
        override val filePath: String = "/tmp/test-benchmark.csv"
        var closed: Boolean = false

        override fun append(stats: PipelineStats, gpuStatus: GpuStatus, modelUsed: String) = Unit

        override fun close() {
            closed = true
        }
    }

    private class FakeTrackingLogger : TrackingLogger {
        override val filePath: String = "/tmp/test-tracking.csv"
        var closed: Boolean = false

        override fun append(frame: DetectionFrame, delegateMode: InferenceMode, modelUsed: String) = Unit

        override fun close() {
            closed = true
        }
    }

    private class FakeFrameProcessor : FrameProcessorGateway {
        private val statsFlow = MutableStateFlow(PipelineStats())
        private val detectionsFlow = MutableStateFlow(DetectionFrame())
        private val poseFlow = MutableStateFlow<LivePoseOverlay?>(null)
        private val switchingFlow = MutableStateFlow(false)
        private val errorFlow = MutableStateFlow<String?>(null)
        private val rotationTelemetryFlow = MutableStateFlow(RotationTelemetry())
        private val poseValidationResultFlow: Flow<PoseFrameResult> = emptyFlow()

        var resetInterpreterCalls: Int = 0
        var lastSetMode: InferenceMode? = null
        var lastSetMaxMissFrames: Int = 10
        var lastProcessNoise: Float = 0f
        var lastMeasurementNoise: Float = 0f
        var submitPoseValidationBatchCalls: Int = 0

        override val stats: StateFlow<PipelineStats> = statsFlow
        override val detections: StateFlow<DetectionFrame> = detectionsFlow
        override val poseResult: StateFlow<LivePoseOverlay?> = poseFlow
        override val isSwitchingMode: StateFlow<Boolean> = switchingFlow
        override val lastError: StateFlow<String?> = errorFlow
        override val rotationTelemetry: StateFlow<RotationTelemetry> = rotationTelemetryFlow
        override val poseValidationResults: Flow<PoseFrameResult> = poseValidationResultFlow

        override fun submitImage(image: ImageProxy) {
            image.close()
        }

        override fun submitPoseValidationBatch(bitmaps: List<android.graphics.Bitmap>) {
            submitPoseValidationBatchCalls += 1
        }

        override fun setInferenceMode(mode: InferenceMode) {
            lastSetMode = mode
            statsFlow.value = statsFlow.value.copy(delegateMode = mode)
        }

        override fun setPoseGatingMode(mode: PoseGatingMode) = Unit

        override fun setPersonSelectionMode(mode: PersonSelectionMode) = Unit

        override fun setTrackerMaxMissFrames(maxMissFrames: Int) {
            lastSetMaxMissFrames = maxMissFrames
        }

        override fun setTrackerNoise(processNoise: Float, measurementNoise: Float) {
            lastProcessNoise = processNoise
            lastMeasurementNoise = measurementNoise
        }

        override fun updateExpectedRotation(
            expectedTargetRotation: Int,
            expectedFrameRotationDegrees: Int,
            source: String
        ) {
            rotationTelemetryFlow.value = rotationTelemetryFlow.value.copy(
                expectedTargetRotation = expectedTargetRotation,
                expectedFrameRotationDegrees = expectedFrameRotationDegrees
            )
        }

        override fun resetInterpreter() {
            resetInterpreterCalls += 1
        }

        override fun shutdown() = Unit

        fun emitPoseOverlay(overlay: LivePoseOverlay?) {
            poseFlow.value = overlay
        }
    }

    private companion object {
        private const val MODEL_A = "yolov8n_saved_model/yolov8n_float16.tflite"
        private const val MODEL_B = "custom_saved_model/custom_float16.tflite"
        private const val NNAPI_AVAILABLE = "NNAPI_AVAILABLE"
        private const val NNAPI_UNAVAILABLE = "NNAPI_UNAVAILABLE: test"
    }

    private data class ViewModelHandle(
        val store: ViewModelStore,
        val viewModel: CameraViewModel
    )
}

@OptIn(ExperimentalCoroutinesApi::class)
class MainDispatcherExtension(
    private val dispatcher: TestDispatcher = UnconfinedTestDispatcher()
) : BeforeEachCallback, AfterEachCallback {
    override fun beforeEach(context: ExtensionContext) {
        Dispatchers.setMain(dispatcher)
    }

    override fun afterEach(context: ExtensionContext) {
        Dispatchers.resetMain()
    }
}
