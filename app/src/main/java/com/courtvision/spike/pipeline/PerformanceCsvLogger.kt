package com.courtvision.spike.pipeline

import android.content.Context
import java.io.File
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

class PerformanceCsvLogger(context: Context) {
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

        val fileName = "phase0-day1-day2-${fileNameFormatter.format(Instant.now())}.csv"
        outputFile = File(outputDir, fileName)
        outputFile.appendText(
            "timestamp,analysis_fps,avg_analyze_ms,p95_analyze_ms,dropped_frames,queue_depth,gpu_status\n"
        )
    }

    val filePath: String
        get() = outputFile.absolutePath

    fun append(stats: PipelineStats, gpuStatus: GpuStatus) {
        val timestamp = timestampFormatter.format(Instant.now())
        val row = buildString {
            append(timestamp)
            append(',')
            append(stats.analysisFps)
            append(',')
            append(formatDouble(stats.avgAnalyzeMs))
            append(',')
            append(formatDouble(stats.p95AnalyzeMs))
            append(',')
            append(stats.droppedFrames)
            append(',')
            append(stats.queueDepth)
            append(',')
            append(gpuStatus.name)
            append('\n')
        }

        synchronized(lock) {
            outputFile.appendText(row)
        }
    }

    private fun formatDouble(value: Double): String = String.format(Locale.US, "%.3f", value)
}
