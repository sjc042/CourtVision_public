package com.courtvision.spike.camera

import android.app.Application
import androidx.camera.core.ImageAnalysis
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.courtvision.spike.pipeline.FrameProcessor
import com.courtvision.spike.pipeline.GpuDelegateProbe
import com.courtvision.spike.pipeline.PerformanceCsvLogger
import com.courtvision.spike.pipeline.SpikeImageAnalyzer
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class CameraViewModel(
    application: Application
) : AndroidViewModel(application) {

    private val frameProcessor = FrameProcessor(viewModelScope)
    private val csvLogger = PerformanceCsvLogger(application)
    private val analyzer = SpikeImageAnalyzer(frameProcessor)

    private val _uiState = MutableStateFlow(
        CameraUiState(
            logFilePath = csvLogger.filePath
        )
    )
    val uiState: StateFlow<CameraUiState> = _uiState.asStateFlow()

    init {
        val probeResult = GpuDelegateProbe.probe()
        _uiState.update { it.copy(gpuProbeResult = probeResult) }

        viewModelScope.launch {
            frameProcessor.stats.collect { stats ->
                _uiState.update { state ->
                    state.copy(stats = stats)
                }
                csvLogger.append(stats, _uiState.value.gpuProbeResult.status)
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

    fun setSimulatedDelayMs(delayMs: Long) {
        frameProcessor.setSimulatedDelayMs(delayMs)
        _uiState.update {
            it.copy(simulatedDelayMs = delayMs)
        }
    }

    override fun onCleared() {
        frameProcessor.shutdown()
        super.onCleared()
    }
}
