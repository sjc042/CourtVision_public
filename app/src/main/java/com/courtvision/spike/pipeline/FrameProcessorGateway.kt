package com.courtvision.spike.pipeline

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow

interface FrameProcessorGateway {
    val stats: StateFlow<PipelineStats>
    val detections: StateFlow<DetectionFrame>
    val poseResult: StateFlow<LivePoseOverlay?>
    val isSwitchingMode: StateFlow<Boolean>
    val lastError: StateFlow<String?>
    val rotationTelemetry: StateFlow<RotationTelemetry>
    val poseValidationResults: Flow<PoseFrameResult>

    fun submitImage(image: ImageProxy)
    fun submitPoseValidationBatch(bitmaps: List<Bitmap>)
    fun setInferenceMode(mode: InferenceMode)
    fun setPoseGatingMode(mode: PoseGatingMode)
    fun setPersonSelectionMode(mode: PersonSelectionMode)
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
