package com.courtvision.spike.camera

import com.courtvision.spike.pipeline.DetectionFrame
import com.courtvision.spike.pipeline.GpuProbeResult
import com.courtvision.spike.pipeline.InferenceMode
import com.courtvision.spike.pipeline.PipelineStats

enum class CameraStatus {
    IDLE,
    RUNNING,
    PERMISSION_REQUIRED,
    ERROR
}

data class CameraUiState(
    val cameraStatus: CameraStatus = CameraStatus.PERMISSION_REQUIRED,
    val hasCameraPermission: Boolean = false,
    val stats: PipelineStats = PipelineStats(),
    val gpuProbeResult: GpuProbeResult = GpuProbeResult(),
    val availableModels: List<String> = emptyList(),
    val selectedModel: String = "",
    val modelConfirmed: Boolean = false,
    val selectedMode: InferenceMode = InferenceMode.CPU,
    val isSwitchingMode: Boolean = false,
    val detectionFrame: DetectionFrame = DetectionFrame(),
    val logFilePath: String = "",
    val lastError: String? = null
)
