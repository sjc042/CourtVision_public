package com.courtvision.spike.camera

import com.courtvision.spike.pipeline.DetectionFrame
import com.courtvision.spike.pipeline.GpuProbeResult
import com.courtvision.spike.pipeline.InferenceMode
import com.courtvision.spike.pipeline.LivePoseOverlay
import com.courtvision.spike.pipeline.PipelineStats
import com.courtvision.spike.pipeline.QnnProbeResult

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
    val qnnProbeResult: QnnProbeResult = QnnProbeResult(),
    val qnnAvailable: Boolean = false,
    val nnApiProbeResult: String = "NNAPI_UNAVAILABLE: not probed",
    val nnApiAvailable: Boolean = false,
    val availableModels: List<String> = emptyList(),
    val selectedModel: String = "",
    val modelConfirmed: Boolean = false,
    val trackerMaxMissFrames: Int = 10,
    val trackerProcessNoise: Float = 1e-2f,
    val trackerMeasurementNoise: Float = 1e-5f,
    val selectedMode: InferenceMode = InferenceMode.CPU,
    val isSwitchingMode: Boolean = false,
    val detectionFrame: DetectionFrame = DetectionFrame(),
    val poseOverlay: LivePoseOverlay? = null,
    val logFilePath: String = "",
    val trackingLogFilePath: String = "",
    val poseValidationRunning: Boolean = false,
    val poseValidationStatus: String = "IDLE",
    val poseValidationOutputPath: String = "",
    val lastError: String? = null
)
