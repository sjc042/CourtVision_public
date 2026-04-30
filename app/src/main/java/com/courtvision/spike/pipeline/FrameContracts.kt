package com.courtvision.spike.pipeline

data class FramePacket(
    val timestampNs: Long,
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val format: Int
)

data class PipelineStats(
    val analysisFps: Int = 0,
    val avgAnalyzeMs: Double = 0.0,
    val p95AnalyzeMs: Double = 0.0,
    val droppedFrames: Long = 0,
    val queueDepth: Int = 0,
    val lastInferenceMs: Double = 0.0,
    val delegateMode: InferenceMode = InferenceMode.CPU,
    val ramMb: Double = 0.0,
    val thermalStatus: String = "UNKNOWN",
    val trackingActive: Boolean = false,
    val trackCx: Double? = null,
    val trackCy: Double? = null,
    val trackVx: Double? = null,
    val trackVy: Double? = null,
    val missStreak: Int = 0,
    val poseSkipped: Boolean = true
)

interface FrameConsumer {
    suspend fun consume(frame: FramePacket)
}

enum class GpuStatus {
    GPU_SUPPORTED,
    GPU_UNSUPPORTED,
    GPU_INIT_FAILED
}

data class GpuProbeResult(
    val status: GpuStatus = GpuStatus.GPU_UNSUPPORTED,
    val reason: String? = null,
    val deviceModel: String = "",
    val apiLevel: Int = 0
)

enum class QnnStatus {
    QNN_SUPPORTED,
    QNN_UNSUPPORTED,
    QNN_INIT_FAILED
}

data class QnnProbeResult(
    val status: QnnStatus = QnnStatus.QNN_UNSUPPORTED,
    val htpFp16Supported: Boolean = false,
    val htpQuantizedSupported: Boolean = false,
    val reason: String? = null,
    val deviceModel: String = "",
    val apiLevel: Int = 0
)

enum class InferenceMode {
    CPU,
    GPU,
    NNAPI,
    QNN_NPU
}

enum class PoseGatingMode {
    EVERY_FRAME_WITH_PERSON,
    SHOOT_CLASS_GATED,
    FSM_GATED
}

enum class PersonSelectionMode {
    HIGHEST_CONFIDENCE,
    REID_TRACKED
}

enum class RotationStallState {
    NONE,
    MISMATCH,
    RECONCILE,
    RECOVERY_REQUESTED
}

data class RotationTelemetry(
    val expectedTargetRotation: Int = -1,
    val expectedFrameRotationDegrees: Int = -1,
    val frameRotationDegrees: Int = -1,
    val rotationMismatchMs: Long = 0L,
    val droppedFramesSnapshot: Long = 0L,
    val stallState: RotationStallState = RotationStallState.NONE
)

data class DetectionBox(
    val classId: Int,
    val label: String,
    val confidence: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

data class TrackedBall(
    val centroidX: Float,
    val centroidY: Float,
    val velocityX: Float,
    val velocityY: Float,
    val isTracked: Boolean,
    val rawBox: DetectionBox?
)

data class DetectionFrame(
    val timestampNs: Long = 0L,
    val sourceWidth: Int = 0,
    val sourceHeight: Int = 0,
    val rotationDegrees: Int = 0,
    val boxes: List<DetectionBox> = emptyList(),
    val trackedBall: TrackedBall? = null,
    val missStreak: Int = 0,
    val emitElapsedRealtimeNanos: Long = 0L
)

data class CropRectNormalized(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

data class LivePoseOverlay(
    val poseResult: PoseResult,
    val cropRectNormalized: CropRectNormalized,
    val emitElapsedRealtimeNanos: Long = 0L
)

fun rotateDetectionBox(
    box: DetectionBox,
    rotationDegrees: Int
): DetectionBox {
    val normalizedRotation = ((rotationDegrees % 360) + 360) % 360

    var displayLeft = box.left
    var displayTop = box.top
    var displayRight = box.right
    var displayBottom = box.bottom

    when (normalizedRotation) {
        90 -> {
            displayLeft = 1f - box.bottom
            displayTop = box.left
            displayRight = 1f - box.top
            displayBottom = box.right
        }
        180 -> {
            displayLeft = 1f - box.right
            displayTop = 1f - box.bottom
            displayRight = 1f - box.left
            displayBottom = 1f - box.top
        }
        270 -> {
            displayLeft = box.top
            displayTop = 1f - box.right
            displayRight = box.bottom
            displayBottom = 1f - box.left
        }
    }

    val clampedLeft = displayLeft.coerceIn(0f, 1f)
    val clampedTop = displayTop.coerceIn(0f, 1f)
    val clampedRight = displayRight.coerceIn(0f, 1f)
    val clampedBottom = displayBottom.coerceIn(0f, 1f)

    return box.copy(
        left = minOf(clampedLeft, clampedRight),
        top = minOf(clampedTop, clampedBottom),
        right = maxOf(clampedLeft, clampedRight),
        bottom = maxOf(clampedTop, clampedBottom)
    )
}
