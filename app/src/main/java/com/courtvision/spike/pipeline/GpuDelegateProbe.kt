package com.courtvision.spike.pipeline

import android.os.Build
import java.nio.MappedByteBuffer
import java.security.MessageDigest
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory

data class SustainedSpeedGpuDelegateConfig(
    val inferencePreference: Int,
    val isPrecisionLossAllowed: Boolean,
    val serializationCacheDir: String?,
    val serializationModelToken: String?
)

internal fun sustainedSpeedGpuDelegateConfig(
    cacheDir: String? = null,
    modelToken: String? = null
): SustainedSpeedGpuDelegateConfig {
    val usableCacheDir = cacheDir?.takeIf { it.isNotBlank() }
    val usableModelToken = modelToken?.takeIf { it.isNotBlank() }
    val serializationEnabled = usableCacheDir != null && usableModelToken != null
    return SustainedSpeedGpuDelegateConfig(
        inferencePreference = GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED,
        isPrecisionLossAllowed = true,
        serializationCacheDir = if (serializationEnabled) usableCacheDir else null,
        serializationModelToken = if (serializationEnabled) usableModelToken else null
    )
}

internal fun computeModelBufferMd5(buffer: MappedByteBuffer): String {
    val duplicate = buffer.duplicate()
    duplicate.position(0)
    val bytes = ByteArray(duplicate.remaining())
    duplicate.get(bytes)
    return MessageDigest.getInstance("MD5")
        .digest(bytes)
        .joinToString(separator = "") { "%02x".format(it) }
}

fun buildSustainedSpeedGpuDelegate(
    cacheDir: String? = null,
    modelToken: String? = null
): GpuDelegate =
    buildSustainedSpeedGpuDelegate(
        sustainedSpeedGpuDelegateConfig(
            cacheDir = cacheDir,
            modelToken = modelToken
        )
    )

internal fun buildSustainedSpeedGpuDelegate(
    config: SustainedSpeedGpuDelegateConfig
): GpuDelegate {
    return try {
        val factoryOptions = GpuDelegateFactory.Options().apply {
            setInferencePreference(config.inferencePreference)
            setPrecisionLossAllowed(config.isPrecisionLossAllowed)
            if (config.serializationCacheDir != null && config.serializationModelToken != null) {
                setSerializationParams(config.serializationCacheDir, config.serializationModelToken)
            }
        }
        GpuDelegate(factoryOptions)
    } catch (_: Throwable) {
        GpuDelegate(
            GpuDelegate.Options().apply {
                setInferencePreference(config.inferencePreference)
                setPrecisionLossAllowed(config.isPrecisionLossAllowed)
            }
        )
    }
}

private fun fallbackProbeConfig(): SustainedSpeedGpuDelegateConfig =
    SustainedSpeedGpuDelegateConfig(
        inferencePreference = GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED,
        isPrecisionLossAllowed = true,
        serializationCacheDir = null,
        serializationModelToken = null
    )

object GpuDelegateProbe {
    fun probe(): GpuProbeResult {
        val model = Build.MODEL ?: "unknown"
        val apiLevel = Build.VERSION.SDK_INT
        val compatibilityList = CompatibilityList()

        if (!compatibilityList.isDelegateSupportedOnThisDevice) {
            return GpuProbeResult(
                status = GpuStatus.GPU_UNSUPPORTED,
                reason = "CompatibilityList reports unsupported delegate",
                deviceModel = model,
                apiLevel = apiLevel
            )
        }

        return try {
            val delegate = buildSustainedSpeedGpuDelegate(fallbackProbeConfig())
            delegate.close()
            GpuProbeResult(
                status = GpuStatus.GPU_SUPPORTED,
                reason = null,
                deviceModel = model,
                apiLevel = apiLevel
            )
        } catch (error: Throwable) {
            GpuProbeResult(
                status = GpuStatus.GPU_INIT_FAILED,
                reason = "${error::class.java.simpleName}: ${error.message ?: "unknown error"}",
                deviceModel = model,
                apiLevel = apiLevel
            )
        }
    }
}
