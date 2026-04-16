package com.courtvision.spike.pipeline

data class PoseImageLandmark(
    val xPx: Float,
    val yPx: Float,
    val zPx: Float,
    val visibilityLogit: Float,
    val presenceLogit: Float,
    val visibility: Float,
    val presence: Float
)

data class PoseWorldLandmark(
    val x: Float,
    val y: Float,
    val z: Float
)

data class PoseStageLatency(
    val preprocessMs: Double,
    val inferenceMs: Double,
    val postprocessMs: Double,
    val totalMs: Double
)

data class PoseResult(
    val imageLandmarks39: List<PoseImageLandmark>,
    val worldLandmarks39: List<PoseWorldLandmark>,
    val imageLandmarks33: List<PoseImageLandmark>,
    val worldLandmarks33: List<PoseWorldLandmark>,
    val posePresenceLogit: Float,
    val posePresence: Float,
    val latency: PoseStageLatency
)

object PoseLandmarkIndex {
    const val LEFT_SHOULDER = 11
    const val RIGHT_SHOULDER = 12
    const val LEFT_ELBOW = 13
    const val RIGHT_ELBOW = 14
    const val LEFT_WRIST = 15
    const val RIGHT_WRIST = 16
    const val LEFT_HIP = 23
    const val RIGHT_HIP = 24
    const val LEFT_KNEE = 25
    const val RIGHT_KNEE = 26
    const val LEFT_ANKLE = 27
    const val RIGHT_ANKLE = 28
}

object PoseTensorContract {
    const val INPUT_SIZE = 256
    const val LANDMARKS_TOTAL = 39
    const val LANDMARKS_CANONICAL = 33
    const val IMAGE_FIELDS = 5
    const val WORLD_FIELDS = 3
    const val IMAGE_OUTPUT_SIZE = LANDMARKS_TOTAL * IMAGE_FIELDS
    const val WORLD_OUTPUT_SIZE = LANDMARKS_TOTAL * WORLD_FIELDS
}

fun PoseResult.worldLandmarkVisibility(index: Int): Float {
    require(index in imageLandmarks33.indices) {
        "Landmark index out of range: $index"
    }
    return imageLandmarks33[index].visibility
}
