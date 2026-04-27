package com.courtvision.spike.pipeline

import android.content.Context
import android.util.Log
import java.io.BufferedWriter
import java.io.File
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

interface PerFramePerfLogger {
    val filePath: String

    fun append(row: PerFramePerfRow)

    fun close()
}

class PerFramePerfCsvLogger private constructor(
    baseOutputDir: File
) : PerFramePerfLogger {
    private val lock = Any()
    private val timestampFormatter =
        DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).withZone(ZoneId.systemDefault())
    private val writerExecutor = Executors.newSingleThreadExecutor()
    private val closed = AtomicBoolean(false)
    private val pendingWrites = AtomicInteger(0)
    private val droppedRows = AtomicInteger(0)

    private val outputDir: File = baseOutputDir
    private val outputFile: File
    private val writer: BufferedWriter
    private var rowsSinceFlush: Int = 0

    constructor(context: Context) : this(
        File(
            context.getExternalFilesDir(null) ?: context.filesDir,
            OUTPUT_DIRECTORY_RELATIVE_PATH
        )
    )

    internal constructor(outputDir: File, useTestOutputDir: Boolean) : this(
        if (useTestOutputDir) outputDir else File(outputDir, OUTPUT_DIRECTORY_RELATIVE_PATH)
    )

    init {
        if (!outputDir.exists()) {
            outputDir.mkdirs()
        }
        val fileName = "phase0-perframe-${FILE_NAME_FORMATTER.format(Instant.now())}.csv"
        outputFile = File(outputDir, fileName)
        writer = outputFile.bufferedWriter()
        writer.write(CSV_HEADER)
        writer.flush()
    }

    override val filePath: String
        get() = outputFile.absolutePath

    override fun append(row: PerFramePerfRow) {
        if (closed.get()) return

        val pending = pendingWrites.incrementAndGet()
        if (pending > MAX_PENDING_ROWS) {
            pendingWrites.decrementAndGet()
            val dropped = droppedRows.incrementAndGet()
            if (dropped % DROP_LOG_INTERVAL == 0) {
                Log.w(TAG, "Dropping per-frame rows under write pressure: dropped=$dropped")
            }
            return
        }

        val rowString = buildCsvRow(row, timestampFormatter)
        writerExecutor.execute {
            try {
                synchronized(lock) {
                    writer.write(rowString)
                    rowsSinceFlush += 1
                    if (rowsSinceFlush >= FLUSH_EVERY_ROWS) {
                        writer.flush()
                        rowsSinceFlush = 0
                    }
                }
            } finally {
                pendingWrites.decrementAndGet()
            }
        }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return

        writerExecutor.execute {
            synchronized(lock) {
                runCatching { writer.flush() }
                runCatching { writer.close() }
            }
        }

        writerExecutor.shutdown()
        runCatching {
            writerExecutor.awaitTermination(2, TimeUnit.SECONDS)
        }
    }

    companion object {
        private const val TAG = "PerFramePerfCsvLogger"
        private const val OUTPUT_DIRECTORY_RELATIVE_PATH = "benchmarks/phase0/combined_pipeline"
        private const val FLUSH_EVERY_ROWS = 64
        private const val MAX_PENDING_ROWS = 1_000
        private const val DROP_LOG_INTERVAL = 100

        private val FILE_NAME_FORMATTER =
            DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss", Locale.US).withZone(ZoneId.systemDefault())

        private const val CSV_HEADER =
            "timestamp,frame_total_ms,yolo_preprocess_ms,yolo_inference_ms,yolo_nms_ms," +
                "pose_crop_ms,pose_preprocess_ms,pose_inference_ms,pose_postprocess_ms," +
                "pose_skipped,ram_mb,thermal_status,fps_1s_window,rotation_degrees," +
                "mode,device,gpu_mode\n"

        internal fun csvHeaderForTest(): String = CSV_HEADER

        internal fun buildCsvRow(
            row: PerFramePerfRow,
            timestampFormatter: DateTimeFormatter =
                DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
                    .withZone(ZoneId.systemDefault())
        ): String {
            val poseCropMs = if (row.poseSkipped) 0.0 else row.poseCropMs
            val posePreprocessMs = if (row.poseSkipped) 0.0 else row.posePreprocessMs
            val poseInferenceMs = if (row.poseSkipped) 0.0 else row.poseInferenceMs
            val posePostprocessMs = if (row.poseSkipped) 0.0 else row.posePostprocessMs

            return buildString {
                append(timestampFormatter.format(Instant.ofEpochMilli(row.timestampMs)))
                append(',')
                append(formatDouble(row.frameTotalMs))
                append(',')
                append(formatDouble(row.yoloPreprocessMs))
                append(',')
                append(formatDouble(row.yoloInferenceMs))
                append(',')
                append(formatDouble(row.yoloNmsMs))
                append(',')
                append(formatDouble(poseCropMs))
                append(',')
                append(formatDouble(posePreprocessMs))
                append(',')
                append(formatDouble(poseInferenceMs))
                append(',')
                append(formatDouble(posePostprocessMs))
                append(',')
                append(row.poseSkipped)
                append(',')
                append(formatDouble(row.ramMb))
                append(',')
                append(sanitizeCsv(row.thermalStatus))
                append(',')
                append(row.fps1sWindow)
                append(',')
                append(row.rotationDegrees)
                append(',')
                append(sanitizeCsv(row.mode))
                append(',')
                append(sanitizeCsv(row.device))
                append(',')
                append(sanitizeCsv(row.gpuMode))
                append('\n')
            }
        }

        private fun formatDouble(value: Double): String = String.format(Locale.US, "%.3f", value)

        private fun sanitizeCsv(value: String): String =
            value.replace(',', ';').replace('\n', ' ').replace('\r', ' ')
    }
}
