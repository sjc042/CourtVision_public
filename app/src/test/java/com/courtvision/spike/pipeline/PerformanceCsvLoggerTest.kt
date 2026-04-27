package com.courtvision.spike.pipeline

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class PerformanceCsvLoggerTest {

    @Test
    fun csvHeader_includesModelUsedColumn() {
        val header = PerformanceCsvLogger.csvHeaderForTest()

        assertTrue(header.contains("model_used"))
        assertTrue(header.contains("tracking_active"))
        assertEquals(
            "timestamp,analysis_fps,avg_analyze_ms,p95_analyze_ms,last_inference_ms,dropped_frames,queue_depth,delegate_mode,ram_mb,thermal_status,model_used,gpu_status,tracking_active,track_cx,track_cy,track_vx,track_vy,miss_streak\n",
            header
        )
    }

    @Test
    fun buildCsvRow_includesModelPathValue() {
        val row = PerformanceCsvLogger.buildCsvRow(
            timestamp = "2026-03-28 09:00:00",
            stats = PipelineStats(
                analysisFps = 24,
                avgAnalyzeMs = 41.2,
                p95AnalyzeMs = 66.8,
                droppedFrames = 3,
                queueDepth = 1,
                lastInferenceMs = 39.7,
                delegateMode = InferenceMode.GPU,
                ramMb = 231.4,
                thermalStatus = "LIGHT",
                trackingActive = true,
                trackCx = 0.45,
                trackCy = 0.55,
                trackVx = 0.12,
                trackVy = -0.31,
                missStreak = 2
            ),
            gpuStatus = GpuStatus.GPU_SUPPORTED,
            modelUsed = "yolo11s_saved_model/yolo11s_float32.tflite"
        )

        assertTrue(row.contains(",yolo11s_saved_model/yolo11s_float32.tflite,"))
        assertTrue(row.contains(",GPU_SUPPORTED,"))
        assertTrue(row.contains(",1,0.450,0.550,0.120,-0.310,2"))
        assertTrue(row.endsWith(",2\n"))
    }

    @Test
    fun buildCsvRow_serializesNnApiDelegateMode() {
        val row = PerformanceCsvLogger.buildCsvRow(
            timestamp = "2026-03-28 09:00:00",
            stats = PipelineStats(
                delegateMode = InferenceMode.NNAPI
            ),
            gpuStatus = GpuStatus.GPU_SUPPORTED,
            modelUsed = "yolov8n_saved_model/yolov8n_float16.tflite"
        )

        assertTrue(row.contains(",NNAPI,"))
    }
}
