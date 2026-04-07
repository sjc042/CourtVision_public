package com.courtvision.spike.pipeline

interface TrackingLogger {
    val filePath: String

    fun append(
        frame: DetectionFrame,
        delegateMode: InferenceMode,
        modelUsed: String
    )

    fun close()
}
