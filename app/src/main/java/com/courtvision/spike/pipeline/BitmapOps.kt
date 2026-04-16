package com.courtvision.spike.pipeline

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color

/**
 * Returns a square bitmap with [source] centered and padded with [padColor].
 * If [source] is already square, returns it unchanged (no allocation).
 *
 * Required by pose_landmark_lite which expects a square ROI; anisotropic resize
 * to 256x256 distorts landmarks for non-square inputs.
 *
 * Caller is responsible for recycling the returned bitmap ONLY when it differs
 * from [source]: `if (padded !== source) padded.recycle()`.
 */
fun squarePadCrop(source: Bitmap, padColor: Int = Color.BLACK): Bitmap {
    if (source.width == source.height) return source

    val maxDim = maxOf(source.width, source.height)
    val out = Bitmap.createBitmap(maxDim, maxDim, Bitmap.Config.ARGB_8888)
    out.eraseColor(padColor)
    Canvas(out).apply {
        val left = (maxDim - source.width) / 2f
        val top = (maxDim - source.height) / 2f
        drawBitmap(source, left, top, null)
    }
    return out
}
