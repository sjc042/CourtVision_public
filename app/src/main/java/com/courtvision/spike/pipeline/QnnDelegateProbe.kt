package com.courtvision.spike.pipeline

import android.os.Build
import com.qualcomm.qti.QnnDelegate

object QnnDelegateProbe {
    fun probe(): QnnProbeResult {
        val model = Build.MODEL ?: "unknown"
        val apiLevel = Build.VERSION.SDK_INT

        if (apiLevel < Build.VERSION_CODES.S) {
            return QnnProbeResult(
                status = QnnStatus.QNN_UNSUPPORTED,
                reason = "QNN requires API 31+ (S); device is API $apiLevel",
                deviceModel = model,
                apiLevel = apiLevel
            )
        }

        return try {
            val htpFp16 =
                QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16)
            val htpQuantized =
                QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_QUANTIZED)
            QnnProbeResult(
                status = if (htpQuantized) {
                    QnnStatus.QNN_SUPPORTED
                } else {
                    QnnStatus.QNN_UNSUPPORTED
                },
                htpFp16Supported = htpFp16,
                htpQuantizedSupported = htpQuantized,
                reason = if (htpQuantized) {
                    null
                } else {
                    "HTP quantized runtime not available on this device"
                },
                deviceModel = model,
                apiLevel = apiLevel
            )
        } catch (error: Throwable) {
            QnnProbeResult(
                status = QnnStatus.QNN_INIT_FAILED,
                reason = "${error::class.java.simpleName}: ${error.message ?: "unknown error"}",
                deviceModel = model,
                apiLevel = apiLevel
            )
        }
    }
}
