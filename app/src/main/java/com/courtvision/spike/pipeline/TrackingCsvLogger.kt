package com.courtvision.spike.pipeline

import android.content.Context
import java.io.File
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class TrackingCsvLogger(
    context: Context
) : TrackingLogger {
    private val lock = Any()
    private val fileNameFormatter =
        DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss", Locale.US).withZone(ZoneId.systemDefault())
    private val writerExecutor = Executors.newSingleThreadExecutor()

    private val outputDir: File = File(
        context.getExternalFilesDir(null) ?: context.filesDir,
        "benchmarks"
    )
    private val outputFile: File

    init {
        if (!outputDir.exists()) {
            // Phase 2: move directory creation off Main so logger init does not do filesystem work on startup.
            outputDir.mkdirs()
        }

        val fileName = "phase0-day4-track-${fileNameFormatter.format(Instant.now())}.csv"
        outputFile = File(outputDir, fileName)
        enqueueWrite(CSV_HEADER)
    }

    override val filePath: String
        get() = outputFile.absolutePath

    override fun append(
        frame: DetectionFrame,
        delegateMode: InferenceMode,
        modelUsed: String
    ) {
        val tracked = frame.trackedBall
        val isTracked = tracked?.isTracked == true
        val row = buildCsvRow(
            timestampNs = frame.timestampNs,
            trackingActive = isTracked,
            trackCx = tracked?.centroidX?.toDouble(),
            trackCy = tracked?.centroidY?.toDouble(),
            trackVx = tracked?.velocityX?.toDouble(),
            trackVy = tracked?.velocityY?.toDouble(),
            missStreak = frame.missStreak,
            delegateMode = delegateMode,
            modelUsed = modelUsed
        )

        enqueueWrite(row)
    }

    override fun close() {
        writerExecutor.shutdown()
        runCatching {
            writerExecutor.awaitTermination(1, TimeUnit.SECONDS)
        }
    }

    private fun enqueueWrite(content: String) {
        writerExecutor.execute {
            synchronized(lock) {
                outputFile.appendText(content)
            }
        }
    }

    companion object {
        private const val CSV_HEADER =
            "timestamp_ns,tracking_active,track_cx,track_cy,track_vx,track_vy,miss_streak,delegate_mode,model_used\n"

        internal fun buildCsvRow(
            timestampNs: Long,
            trackingActive: Boolean,
            trackCx: Double?,
            trackCy: Double?,
            trackVx: Double?,
            trackVy: Double?,
            missStreak: Int,
            delegateMode: InferenceMode,
            modelUsed: String
        ): String {
            return buildString {
                append(timestampNs)
                append(',')
                append(if (trackingActive) 1 else 0)
                append(',')
                append(formatNullableDouble(trackCx))
                append(',')
                append(formatNullableDouble(trackCy))
                append(',')
                append(formatNullableDouble(trackVx))
                append(',')
                append(formatNullableDouble(trackVy))
                append(',')
                append(missStreak)
                append(',')
                append(delegateMode.name)
                append(',')
                append(modelUsed)
                append('\n')
            }
        }

        private fun formatNullableDouble(value: Double?): String {
            return if (value == null) "" else String.format(Locale.US, "%.6f", value)
        }
    }
}
