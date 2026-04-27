package com.courtvision.spike.pipeline

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.tensorflow.lite.DataType

class PoseLandmarkInterpreterTest {

    @Test
    fun decodeImageLandmarks_195floats_mapsTo39Landmarks() {
        val raw = FloatArray(PoseTensorContract.IMAGE_OUTPUT_SIZE) { it.toFloat() / 10f }

        val decoded = PoseLandmarkInterpreter.decodeImageLandmarks(raw)

        assertEquals(PoseTensorContract.LANDMARKS_TOTAL, decoded.size)
        assertEquals(raw[0], decoded[0].xPx, 1e-6f)
        assertEquals(raw[1], decoded[0].yPx, 1e-6f)
        assertEquals(raw[2], decoded[0].zPx, 1e-6f)
    }

    @Test
    fun decodeWorldLandmarks_117floats_mapsTo39Landmarks() {
        val raw = FloatArray(PoseTensorContract.WORLD_OUTPUT_SIZE) { it.toFloat() / 10f }

        val decoded = PoseLandmarkInterpreter.decodeWorldLandmarks(raw)

        assertEquals(PoseTensorContract.LANDMARKS_TOTAL, decoded.size)
        assertEquals(raw[0], decoded[0].x, 1e-6f)
        assertEquals(raw[1], decoded[0].y, 1e-6f)
        assertEquals(raw[2], decoded[0].z, 1e-6f)
    }

    @Test
    fun first33CanonicalSubset_canBeDerivedFromDecoded39() {
        val raw = FloatArray(PoseTensorContract.IMAGE_OUTPUT_SIZE) { it.toFloat() }
        val decoded = PoseLandmarkInterpreter.decodeImageLandmarks(raw)

        val subset = decoded.take(PoseTensorContract.LANDMARKS_CANONICAL)

        assertEquals(PoseTensorContract.LANDMARKS_CANONICAL, subset.size)
    }

    @Test
    fun sigmoidTransformsLogitsForVisibilityAndPresence() {
        val positive = PoseLandmarkInterpreter.sigmoid(2f)
        val negative = PoseLandmarkInterpreter.sigmoid(-2f)

        assertTrue(positive > 0.6f)
        assertTrue(negative < 0.5f)
    }

    @Test
    fun resolveOutputIndices_matchesByShape_notByFixedTensorId() {
        val outputShapes = listOf(
            intArrayOf(1, 1), // presence
            intArrayOf(1, 117), // world
            intArrayOf(1, 64, 64, 39), // ignored heatmap
            intArrayOf(1, 195) // image
        )

        val indices = PoseLandmarkInterpreter.resolveOutputIndices(outputShapes)

        assertEquals(3, indices.image)
        assertEquals(0, indices.presence)
        assertEquals(1, indices.world)
    }

    @Test
    fun resolveOutputIndices_throwsWhenExpectedOutputMissing() {
        val outputShapes = listOf(
            intArrayOf(1, 1),
            intArrayOf(1, 64, 64, 39)
        )

        assertThrows<IllegalArgumentException> {
            PoseLandmarkInterpreter.resolveOutputIndices(outputShapes)
        }
    }

    @Test
    fun validateInputTensor_throwsOnWrongShape() {
        assertThrows<IllegalArgumentException> {
            PoseLandmarkInterpreter.validateInputTensor(
                inputShape = intArrayOf(1, 224, 224, 3),
                inputType = DataType.FLOAT32
            )
        }
    }

    @Test
    fun resolveRuntimeConfig_cpuFallback_usesFourThreads() {
        val config = PoseLandmarkInterpreter.resolveRuntimeConfig(useGpu = false)

        assertEquals(false, config.useGpu)
        assertEquals(4, config.cpuThreads)
    }

    @Test
    fun worldLandmarkVisibility_readsFromImageLandmarks33() {
        val imageLandmarks = (0 until PoseTensorContract.LANDMARKS_TOTAL).map { index ->
            PoseImageLandmark(
                xPx = index.toFloat(),
                yPx = index.toFloat(),
                zPx = index.toFloat(),
                visibilityLogit = 0f,
                presenceLogit = 0f,
                visibility = index / 100f,
                presence = 1f
            )
        }
        val worldLandmarks = (0 until PoseTensorContract.LANDMARKS_TOTAL).map {
            PoseWorldLandmark(x = it.toFloat(), y = it.toFloat(), z = it.toFloat())
        }
        val result = PoseResult(
            imageLandmarks39 = imageLandmarks,
            worldLandmarks39 = worldLandmarks,
            imageLandmarks33 = imageLandmarks.take(PoseTensorContract.LANDMARKS_CANONICAL),
            worldLandmarks33 = worldLandmarks.take(PoseTensorContract.LANDMARKS_CANONICAL),
            posePresenceLogit = 0f,
            posePresence = 0.5f,
            latency = PoseStageLatency(0.0, 0.0, 0.0, 0.0)
        )

        assertEquals(0.15f, result.worldLandmarkVisibility(PoseLandmarkIndex.LEFT_WRIST), 1e-6f)
    }

    @Test
    fun worldLandmarkVisibility_throwsWhenIndexOutOfRange() {
        val imageLandmarks = (0 until PoseTensorContract.LANDMARKS_CANONICAL).map { index ->
            PoseImageLandmark(
                xPx = index.toFloat(),
                yPx = index.toFloat(),
                zPx = index.toFloat(),
                visibilityLogit = 0f,
                presenceLogit = 0f,
                visibility = 0.8f,
                presence = 1f
            )
        }
        val worldLandmarks = (0 until PoseTensorContract.LANDMARKS_CANONICAL).map {
            PoseWorldLandmark(x = it.toFloat(), y = it.toFloat(), z = it.toFloat())
        }
        val result = PoseResult(
            imageLandmarks39 = imageLandmarks,
            worldLandmarks39 = worldLandmarks,
            imageLandmarks33 = imageLandmarks,
            worldLandmarks33 = worldLandmarks,
            posePresenceLogit = 0f,
            posePresence = 0.5f,
            latency = PoseStageLatency(0.0, 0.0, 0.0, 0.0)
        )

        assertThrows<IllegalArgumentException> {
            result.worldLandmarkVisibility(PoseTensorContract.LANDMARKS_CANONICAL)
        }
    }
}
