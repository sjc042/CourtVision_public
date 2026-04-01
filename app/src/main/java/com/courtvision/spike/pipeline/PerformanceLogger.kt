package com.courtvision.spike.pipeline

interface PerformanceLogger {
    val filePath: String

    fun append(stats: PipelineStats, gpuStatus: GpuStatus, modelUsed: String)
}
