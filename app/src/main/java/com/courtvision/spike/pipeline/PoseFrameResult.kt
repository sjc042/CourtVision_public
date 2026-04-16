package com.courtvision.spike.pipeline

data class PoseFrameResult(
    val sourceIndex: Int,
    val sourceName: String,
    val yoloPreprocessMs: Double,
    val yoloInferenceMs: Double,
    val yoloNmsMs: Double,
    val poseCropMs: Double,
    val posePreprocessMs: Double,
    val poseInferenceMs: Double,
    val posePostprocessMs: Double,
    val frameTotalMs: Double,
    val posePresence: Float,
    val visibleJoints33: Int,
    val decodedLandmarks39: Int,
    val imageLandmarks33: List<PoseImageLandmark>,
    val status: String,
    val error: String?,
    val isLast: Boolean
)
