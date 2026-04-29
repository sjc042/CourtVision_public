package com.courtvision.spike.pipeline

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.tensorflow.lite.gpu.GpuDelegate

class GpuDelegateProbeTest {

    @Test
    fun sustainedSpeedGpuDelegateConfig_enablesSustainedSpeedAndPrecisionLoss() {
        val config = sustainedSpeedGpuDelegateConfig()

        assertEquals(
            GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED,
            config.inferencePreference
        )
        assertTrue(config.isPrecisionLossAllowed)
    }
}
