package com.courtvision.spike.camera

import android.app.Application
import android.os.Build
import android.os.PowerManager
import androidx.camera.core.ImageAnalysis
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.courtvision.spike.pipeline.FrameProcessor
import com.courtvision.spike.pipeline.FrameProcessorGateway
import com.courtvision.spike.pipeline.GpuDelegateProbe
import com.courtvision.spike.pipeline.GpuProbeResult
import com.courtvision.spike.pipeline.DetectionFrame
import com.courtvision.spike.pipeline.InferenceMode
import com.courtvision.spike.pipeline.NnApiDelegateProbe
import com.courtvision.spike.pipeline.PerformanceCsvLogger
import com.courtvision.spike.pipeline.PerformanceLogger
import com.courtvision.spike.pipeline.SpikeImageAnalyzer
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class CameraViewModel(
    application: Application
) : AndroidViewModel(application) {

    private val overrides = testOverrides
    private val availableModelPaths: List<String> = overrides?.modelPaths ?: discoverModelAssets()
    @Volatile
    private var selectedModelPath: String =
        overrides?.initialModelPath
            ?.takeIf { availableModelPaths.contains(it) }
            ?: chooseInitialModel(availableModelPaths)

    private val frameProcessor: FrameProcessorGateway =
        overrides?.frameProcessor ?: FrameProcessor(
            scope = viewModelScope,
            modelBufferProvider = ::loadModelBuffer,
            thermalStatusProvider = ::readThermalStatus
        )
    private val performanceLogger: PerformanceLogger =
        overrides?.performanceLogger ?: PerformanceCsvLogger(application)

    private val _uiState = MutableStateFlow(
        CameraUiState(
            availableModels = availableModelPaths,
            selectedModel = selectedModelPath,
            logFilePath = performanceLogger.filePath
        )
    )
    private val analyzer = SpikeImageAnalyzer(frameProcessor) {
        _uiState.value.modelConfirmed
    }
    val uiState: StateFlow<CameraUiState> = _uiState.asStateFlow()

    @Volatile
    private var cachedModelBuffer: MappedByteBuffer? = null

    init {
        // Phase 0 decision: keep probes synchronous in init for deterministic startup behavior.
        // Phase 2 note: move delegate probes to Dispatchers.Default to avoid any JNI work on Main.
        val probeResult = overrides?.gpuProbeResult ?: GpuDelegateProbe.probe()
        val nnApiResult = overrides?.nnApiProbeResult ?: NnApiDelegateProbe.probe()
        _uiState.update {
            it.copy(
                gpuProbeResult = probeResult,
                nnApiProbeResult = nnApiResult,
                nnApiAvailable = nnApiResult.startsWith("NNAPI_AVAILABLE")
            )
        }

        viewModelScope.launch {
            frameProcessor.stats.collect { stats ->
                _uiState.update { state ->
                    state.copy(
                        stats = stats,
                        selectedMode = stats.delegateMode
                    )
                }
                performanceLogger.append(
                    stats = stats,
                    gpuStatus = _uiState.value.gpuProbeResult.status,
                    modelUsed = selectedModelPath
                )
            }
        }

        viewModelScope.launch {
            frameProcessor.detections.collect { frame ->
                _uiState.update { state ->
                    if (state.modelConfirmed) {
                        state.copy(detectionFrame = frame)
                    } else {
                        state.copy(detectionFrame = DetectionFrame())
                    }
                }
            }
        }

        viewModelScope.launch {
            frameProcessor.isSwitchingMode.collect { switching ->
                _uiState.update { state ->
                    state.copy(isSwitchingMode = switching)
                }
            }
        }

        viewModelScope.launch {
            frameProcessor.lastError.collect { pipelineError ->
                if (pipelineError != null) {
                    _uiState.update { state ->
                        state.copy(lastError = pipelineError)
                    }
                }
            }
        }
    }

    fun imageAnalyzer(): ImageAnalysis.Analyzer = analyzer

    fun onCameraPermissionResult(granted: Boolean) {
        _uiState.update { state ->
            val nextStatus = when {
                !granted -> CameraStatus.PERMISSION_REQUIRED
                state.cameraStatus == CameraStatus.ERROR -> CameraStatus.ERROR
                else -> state.cameraStatus
            }
            state.copy(
                hasCameraPermission = granted,
                cameraStatus = nextStatus
            )
        }
    }

    fun onCameraStarted() {
        _uiState.update {
            it.copy(
                cameraStatus = CameraStatus.RUNNING,
                lastError = null
            )
        }
    }

    fun onCameraStopped() {
        _uiState.update {
            if (it.hasCameraPermission) {
                it.copy(cameraStatus = CameraStatus.IDLE)
            } else {
                it.copy(cameraStatus = CameraStatus.PERMISSION_REQUIRED)
            }
        }
    }

    fun onCameraError(message: String) {
        _uiState.update {
            it.copy(
                cameraStatus = CameraStatus.ERROR,
                lastError = message
            )
        }
    }

    fun setInferenceMode(mode: InferenceMode) {
        frameProcessor.setInferenceMode(mode)
        _uiState.update {
            it.copy(
                selectedMode = mode,
                isSwitchingMode = true
            )
        }
    }

    fun setModel(modelPath: String) {
        if (isModelSelectionLocked()) return
        if (!availableModelPaths.contains(modelPath)) return
        if (modelPath == selectedModelPath) return

        selectedModelPath = modelPath
        cachedModelBuffer = null
        frameProcessor.resetInterpreter()

        _uiState.update {
            it.copy(
                selectedModel = modelPath,
                detectionFrame = DetectionFrame(),
                lastError = null
            )
        }
    }

    fun confirmModel() {
        _uiState.update { state ->
            if (state.modelConfirmed) {
                state
            } else {
                state.copy(modelConfirmed = true)
            }
        }
    }

    fun restartSession() {
        cachedModelBuffer = null
        frameProcessor.resetInterpreter()
        _uiState.update { state ->
            state.copy(
                modelConfirmed = false,
                detectionFrame = DetectionFrame(),
                isSwitchingMode = false,
                lastError = null
            )
        }
    }

    override fun onCleared() {
        frameProcessor.shutdown()
        super.onCleared()
    }

    private fun isModelSelectionLocked(): Boolean = _uiState.value.modelConfirmed

    private fun loadModelBuffer(): MappedByteBuffer {
        cachedModelBuffer?.let { return it }
        synchronized(this) {
            cachedModelBuffer?.let { return it }
            val selectedPath = selectedModelPath
            val mapped = getApplication<Application>().assets.openFd(selectedPath).use { assetFile ->
                FileInputStream(assetFile.fileDescriptor).channel.use { channel ->
                    channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        assetFile.startOffset,
                        assetFile.declaredLength
                    )
                }
            }
            cachedModelBuffer = mapped
            return mapped
        }
    }

    private fun discoverModelAssets(): List<String> {
        val assetManager = getApplication<Application>().assets
        val candidates = mutableSetOf<String>()
        val rootEntries = assetManager.list("").orEmpty()

        rootEntries.forEach { entry ->
            if (isSupportedModelAsset(entry)) {
                candidates.add(entry)
            }

            if (entry.endsWith("_saved_model")) {
                val dirEntries = assetManager.list(entry).orEmpty()
                dirEntries.forEach { fileName ->
                    val candidatePath = "$entry/$fileName"
                    if (isSupportedModelAsset(candidatePath)) {
                        candidates.add(candidatePath)
                    }
                }
            }
        }

        val filtered = candidates
            .asSequence()
            .sorted()
            .toList()

        return if (filtered.isEmpty()) {
            listOf(DEFAULT_MODEL_ASSET_PATH)
        } else {
            filtered
        }
    }

    private fun isSupportedModelAsset(path: String): Boolean {
        return path.endsWith("_float16.tflite") || path.endsWith("_float32.tflite")
    }

    private fun chooseInitialModel(models: List<String>): String {
        return when {
            models.contains(DEFAULT_MODEL_ASSET_PATH) -> DEFAULT_MODEL_ASSET_PATH
            models.isNotEmpty() -> models.first()
            else -> DEFAULT_MODEL_ASSET_PATH
        }
    }

    private fun readThermalStatus(): String {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            return "UNKNOWN"
        }
        val powerManager = getApplication<Application>().getSystemService(PowerManager::class.java)
            ?: return "UNKNOWN"
        return when (powerManager.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> "NONE"
            PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT"
            PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE"
            PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
            PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN"
            else -> "UNKNOWN"
        }
    }

    companion object {
        private const val DEFAULT_MODEL_ASSET_PATH = "yolov8n_saved_model/yolov8n_float16.tflite"

        @Volatile
        internal var testOverrides: TestOverrides? = null
    }

    internal data class TestOverrides(
        val frameProcessor: FrameProcessorGateway? = null,
        val performanceLogger: PerformanceLogger? = null,
        val modelPaths: List<String>? = null,
        val initialModelPath: String? = null,
        val gpuProbeResult: GpuProbeResult? = null,
        val nnApiProbeResult: String? = null
    )
}
