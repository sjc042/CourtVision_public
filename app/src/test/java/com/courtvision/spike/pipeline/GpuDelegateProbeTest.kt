package com.courtvision.spike.pipeline

import java.io.File
import java.io.RandomAccessFile
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.tensorflow.lite.gpu.GpuDelegateFactory

class GpuDelegateProbeTest {

    @Test
    fun sustainedSpeedGpuDelegateConfig_enablesSustainedSpeedAndPrecisionLoss() {
        val config = sustainedSpeedGpuDelegateConfig()

        assertEquals(
            GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED,
            config.inferencePreference
        )
        assertTrue(config.isPrecisionLossAllowed)
    }

    @Test
    fun sustainedSpeedGpuDelegateConfig_includesSerializationOnlyWhenDirAndTokenPresent() {
        val enabled = sustainedSpeedGpuDelegateConfig(
            cacheDir = "/tmp/cache",
            modelToken = "model-token"
        )
        val missingToken = sustainedSpeedGpuDelegateConfig(
            cacheDir = "/tmp/cache",
            modelToken = null
        )
        val blankCacheDir = sustainedSpeedGpuDelegateConfig(
            cacheDir = " ",
            modelToken = "model-token"
        )

        assertEquals("/tmp/cache", enabled.serializationCacheDir)
        assertEquals("model-token", enabled.serializationModelToken)
        assertNull(missingToken.serializationCacheDir)
        assertNull(missingToken.serializationModelToken)
        assertNull(blankCacheDir.serializationCacheDir)
        assertNull(blankCacheDir.serializationModelToken)
    }

    @Test
    fun computeModelBufferMd5_isStableAndPositionSafe() {
        val buffer = createMappedByteBuffer(byteArrayOf(1, 2, 3, 4, 5))
        val expected = computeModelBufferMd5(buffer)

        buffer.position(3)
        val actual = computeModelBufferMd5(buffer)

        assertEquals(expected, actual)
        assertEquals(3, buffer.position())
    }

    private fun createMappedByteBuffer(bytes: ByteArray): MappedByteBuffer {
        val file = File.createTempFile("gpu_delegate_probe", ".bin")
        file.deleteOnExit()
        RandomAccessFile(file, "rw").use { raf ->
            raf.write(bytes)
            raf.channel.use { channel ->
                return channel.map(FileChannel.MapMode.READ_ONLY, 0, bytes.size.toLong())
            }
        }
    }
}
