package com.courtvision.spike.pipeline

import androidx.camera.core.ImageProxy
import kotlinx.coroutines.flow.StateFlow

interface FrameProcessorGateway {
    val stats: StateFlow<PipelineStats>
    val detections: StateFlow<DetectionFrame>
    val isSwitchingMode: StateFlow<Boolean>
    val lastError: StateFlow<String?>
    val rotationTelemetry: StateFlow<RotationTelemetry>

    fun submitImage(image: ImageProxy)
    fun setInferenceMode(mode: InferenceMode)
    fun setTrackerMaxMissFrames(maxMissFrames: Int)
    fun setTrackerNoise(processNoise: Float, measurementNoise: Float)
    fun updateExpectedRotation(
        expectedTargetRotation: Int,
        expectedFrameRotationDegrees: Int,
        source: String
    )
    fun resetInterpreter()
    fun shutdown()
}
