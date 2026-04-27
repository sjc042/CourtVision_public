package com.courtvision.spike.pipeline

import java.nio.file.Path
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir

class PerFramePerfCsvLoggerTest {

    @TempDir
    lateinit var tempDir: Path

    @Test
    fun csvHeader_matchesStep7Schema() {
        assertEquals(
            "timestamp,frame_total_ms,yolo_preprocess_ms,yolo_inference_ms,yolo_nms_ms," +
                "pose_crop_ms,pose_preprocess_ms,pose_inference_ms,pose_postprocess_ms," +
                "pose_skipped,ram_mb,thermal_status,fps_1s_window,rotation_degrees,mode,device,gpu_mode\n",
            PerFramePerfCsvLogger.csvHeaderForTest()
        )
    }

    @Test
    fun buildCsvRow_poseSkippedTrue_zeroesPoseColumns() {
        val row = PerFramePerfRow(
            timestampMs = 1_714_000_000_123L,
            frameTotalMs = 31.25,
            yoloPreprocessMs = 3.2,
            yoloInferenceMs = 12.8,
            yoloNmsMs = 1.1,
            poseCropMs = 2.2,
            posePreprocessMs = 3.3,
            poseInferenceMs = 4.4,
            posePostprocessMs = 5.5,
            poseSkipped = true,
            ramMb = 234.5,
            thermalStatus = "LIGHT",
            fps1sWindow = 29,
            rotationDegrees = 180,
            mode = "sequential_yolo_pose",
            device = "SM-S906U1",
            gpuMode = "GPU"
        )

        val csv = PerFramePerfCsvLogger.buildCsvRow(row)

        assertTrue(csv.contains(",0.000,0.000,0.000,0.000,true,"))
        assertTrue(csv.contains(",29,180,sequential_yolo_pose,SM-S906U1,GPU"))
    }

    @Test
    fun close_flushesPendingWrites() {
        val logger = PerFramePerfCsvLogger(tempDir.toFile(), useTestOutputDir = true)
        logger.append(sampleRow(timestampMs = 1_714_000_000_200L))
        logger.close()

        val lines = tempDir.toFile().walkTopDown()
            .first { it.isFile && it.name.startsWith("phase0-perframe-") }
            .readLines()

        assertEquals(2, lines.size)
        assertTrue(lines[1].contains("sequential_yolo_pose"))
    }

    @Test
    fun bufferedWriter_survivesLargeAppendBurst() {
        val logger = PerFramePerfCsvLogger(tempDir.toFile(), useTestOutputDir = true)
        repeat(1_000) { index ->
            logger.append(sampleRow(timestampMs = 1_714_000_000_300L + index))
        }
        logger.close()

        val lines = tempDir.toFile().walkTopDown()
            .first { it.isFile && it.name.startsWith("phase0-perframe-") }
            .readLines()

        assertEquals(1_001, lines.size) // header + 1000 rows
    }

    private fun sampleRow(timestampMs: Long): PerFramePerfRow {
        return PerFramePerfRow(
            timestampMs = timestampMs,
            frameTotalMs = 25.5,
            yoloPreprocessMs = 2.1,
            yoloInferenceMs = 10.5,
            yoloNmsMs = 1.0,
            poseCropMs = 1.5,
            posePreprocessMs = 1.2,
            poseInferenceMs = 4.7,
            posePostprocessMs = 0.9,
            poseSkipped = false,
            ramMb = 210.0,
            thermalStatus = "NONE",
            fps1sWindow = 30,
            rotationDegrees = 0,
            mode = "sequential_yolo_pose",
            device = "SM-S906U1",
            gpuMode = "GPU"
        )
    }
}
