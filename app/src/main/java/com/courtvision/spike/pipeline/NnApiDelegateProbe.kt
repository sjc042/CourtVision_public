package com.courtvision.spike.pipeline

import android.os.Build
import org.tensorflow.lite.nnapi.NnApiDelegate

object NnApiDelegateProbe {
    fun probe(): String {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O_MR1) {
            return "NNAPI_UNAVAILABLE: requires API 27+"
        }

        return try {
            val delegate = NnApiDelegate()
            delegate.close()
            "NNAPI_AVAILABLE"
        } catch (error: Throwable) {
            "NNAPI_UNAVAILABLE: ${error.message ?: "unknown error"}"
        }
    }
}
