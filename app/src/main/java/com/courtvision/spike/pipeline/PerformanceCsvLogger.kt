package com.courtvision.spike.pipeline

import android.content.Context
import java.io.File
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

class PerformanceCsvLogger(
    context: Context
) : PerformanceLogger {
    private val lock = Any()
    private val timestampFormatter =
        DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss", Locale.US).withZone(ZoneId.systemDefault())
    private val fileNameFormatter =
        DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss", Locale.US).withZone(ZoneId.systemDefault())

    private val outputDir: File = File(
        context.getExternalFilesDir(null) ?: context.filesDir,
        "benchmarks"
    )
    private val outputFile: File

    init {
        if (!outputDir.exists()) {
            outputDir.mkdirs()
        }

        val fileName = "phase0-day3-${fileNameFormatter.format(Instant.now())}.csv"
        outputFile = File(outputDir, fileName)
        outputFile.appendText(CSV_HEADER)
    }

    override val filePath: String
        get() = outputFile.absolutePath

    override fun append(stats: PipelineStats, gpuStatus: GpuStatus, modelUsed: String) {
        val timestamp = timestampFormatter.format(Instant.now())
        val row = buildCsvRow(
            timestamp = timestamp,
            stats = stats,
            gpuStatus = gpuStatus,
            modelUsed = modelUsed
        )

        synchronized(lock) {
            outputFile.appendText(row)
        }
    }

    companion object {
        private const val CSV_HEADER =
            "timestamp,analysis_fps,avg_analyze_ms,p95_analyze_ms,last_inference_ms,dropped_frames,queue_depth,delegate_mode,ram_mb,thermal_status,model_used,gpu_status\n"

        internal fun csvHeaderForTest(): String = CSV_HEADER

        internal fun buildCsvRow(
            timestamp: String,
            stats: PipelineStats,
            gpuStatus: GpuStatus,
            modelUsed: String
        ): String {
            return buildString {
                append(timestamp)
                append(',')
                append(stats.analysisFps)
                append(',')
                append(formatDouble(stats.avgAnalyzeMs))
                append(',')
                append(formatDouble(stats.p95AnalyzeMs))
                append(',')
                append(formatDouble(stats.lastInferenceMs))
                append(',')
                append(stats.droppedFrames)
                append(',')
                append(stats.queueDepth)
                append(',')
                append(stats.delegateMode.name)
                append(',')
                append(formatDouble(stats.ramMb))
                append(',')
                append(stats.thermalStatus)
                append(',')
                append(modelUsed)
                append(',')
                append(gpuStatus.name)
                append('\n')
            }
        }

        private fun formatDouble(value: Double): String = String.format(Locale.US, "%.3f", value)
    }
}
