package com.courtvision.spike.pipeline

import androidx.camera.core.ImageProxy
import kotlinx.coroutines.flow.StateFlow

interface FrameProcessorGateway {
    val stats: StateFlow<PipelineStats>
    val detections: StateFlow<DetectionFrame>
    val isSwitchingMode: StateFlow<Boolean>
    val lastError: StateFlow<String?>

    fun submitImage(image: ImageProxy)
    fun setInferenceMode(mode: InferenceMode)
    fun resetInterpreter()
    fun shutdown()
}
