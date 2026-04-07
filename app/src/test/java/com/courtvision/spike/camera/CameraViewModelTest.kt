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
import com.courtvision.spike.pipeline.PerformanceLogger
import com.courtvision.spike.pipeline.PipelineStats
import com.courtvision.spike.pipeline.RotationTelemetry
import com.courtvision.spike.pipeline.TrackingLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.test.TestDispatcher
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestWatcher
import org.junit.runner.Description

@OptIn(ExperimentalCoroutinesApi::class)
class CameraViewModelTest {

    @get:Rule
    val mainDispatcherRule = MainDispatcherRule()

    @After
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
        private val switchingFlow = MutableStateFlow(false)
        private val errorFlow = MutableStateFlow<String?>(null)
        private val rotationTelemetryFlow = MutableStateFlow(RotationTelemetry())

        var resetInterpreterCalls: Int = 0
        var lastSetMode: InferenceMode? = null
        var lastSetMaxMissFrames: Int = 10
        var lastProcessNoise: Float = 0f
        var lastMeasurementNoise: Float = 0f

        override val stats: StateFlow<PipelineStats> = statsFlow
        override val detections: StateFlow<DetectionFrame> = detectionsFlow
        override val isSwitchingMode: StateFlow<Boolean> = switchingFlow
        override val lastError: StateFlow<String?> = errorFlow
        override val rotationTelemetry: StateFlow<RotationTelemetry> = rotationTelemetryFlow

        override fun submitImage(image: ImageProxy) {
            image.close()
        }

        override fun setInferenceMode(mode: InferenceMode) {
            lastSetMode = mode
            statsFlow.value = statsFlow.value.copy(delegateMode = mode)
        }

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
class MainDispatcherRule(
    private val dispatcher: TestDispatcher = UnconfinedTestDispatcher()
) : TestWatcher() {
    override fun starting(description: Description) {
        Dispatchers.setMain(dispatcher)
    }

    override fun finished(description: Description) {
        Dispatchers.resetMain()
    }
}
