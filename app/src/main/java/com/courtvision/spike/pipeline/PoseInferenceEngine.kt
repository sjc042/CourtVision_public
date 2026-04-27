package com.courtvision.spike.pipeline

import android.graphics.Bitmap

interface PoseInferenceEngine : AutoCloseable {
    fun infer(cropBitmap: Bitmap): PoseResult
}
