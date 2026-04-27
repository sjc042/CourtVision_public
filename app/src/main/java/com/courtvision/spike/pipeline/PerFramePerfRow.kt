package com.courtvision.spike.pipeline

data class PerFramePerfRow(
    val timestampMs: Long,
    val frameTotalMs: Double,
    val yoloPreprocessMs: Double,
    val yoloInferenceMs: Double,
    val yoloNmsMs: Double,
    val poseCropMs: Double,
    val posePreprocessMs: Double,
    val poseInferenceMs: Double,
    val posePostprocessMs: Double,
    val poseSkipped: Boolean,
    val ramMb: Double,
    val thermalStatus: String,
    val fps1sWindow: Int,
    val rotationDegrees: Int,
    val mode: String,
    val device: String,
    val gpuMode: String
)
