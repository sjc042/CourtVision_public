package com.courtvision.spike.pipeline

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

class SpikeImageAnalyzer(
    private val frameProcessor: FrameProcessorGateway,
    private val isInferenceEnabled: () -> Boolean = { true }
) : ImageAnalysis.Analyzer {

    override fun analyze(image: ImageProxy) {
        if (!isInferenceEnabled()) {
            image.close()
            return
        }
        frameProcessor.submitImage(image)
    }
}
