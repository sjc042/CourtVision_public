package com.courtvision.spike.camera

import com.courtvision.spike.pipeline.GpuProbeResult
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
    val simulatedDelayMs: Long = 0,
    val logFilePath: String = "",
    val lastError: String? = null
)
