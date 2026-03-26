package com.courtvision.spike.pipeline

import android.os.Build
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate

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
            val options = compatibilityList.bestOptionsForThisDevice
            val delegate = GpuDelegate(options)
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
