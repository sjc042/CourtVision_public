package com.courtvision.spike.pipeline

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

class SpikeImageAnalyzer(
    private val frameProcessor: FrameProcessor
) : ImageAnalysis.Analyzer {

    override fun analyze(image: ImageProxy) {
        val frame = FramePacket(
            timestampNs = image.imageInfo.timestamp,
            width = image.width,
            height = image.height,
            rotationDegrees = image.imageInfo.rotationDegrees,
            format = image.format
        )

        frameProcessor.submitFrame(frame)
        image.close()
    }
}
