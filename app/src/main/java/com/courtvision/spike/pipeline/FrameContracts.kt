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
    val queueDepth: Int = 0
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
