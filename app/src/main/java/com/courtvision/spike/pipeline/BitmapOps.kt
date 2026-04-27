package com.courtvision.spike.pipeline

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix

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

data class PersonCrop(
    val bitmap: Bitmap,
    val cropRectNormalized: CropRectNormalized
)

fun rotateBitmapForDisplay(source: Bitmap, rotationDegrees: Int): Bitmap {
    val normalizedRotation = ((rotationDegrees % 360) + 360) % 360
    if (normalizedRotation == 0) return source

    val matrix = Matrix().apply {
        postRotate(normalizedRotation.toFloat())
    }
    return Bitmap.createBitmap(
        source,
        0,
        0,
        source.width,
        source.height,
        matrix,
        true
    )
}

/**
 * Crops a square region centered on [personBox], clamps to frame bounds, then scales to pose input size.
 *
 * Caller owns [Bitmap.recycle] on the returned bitmap.
 */
fun squarePadCrop(
    source: Bitmap,
    personBox: DetectionBox,
    marginFactor: Float = 1.25f
): PersonCrop {
    require(marginFactor > 0f) { "marginFactor must be > 0, was $marginFactor" }

    val width = source.width.toFloat()
    val height = source.height.toFloat()
    val centerX = (personBox.left + personBox.right) * 0.5f * width
    val centerY = (personBox.top + personBox.bottom) * 0.5f * height
    val halfSide = maxOf(
        (personBox.right - personBox.left) * marginFactor * width,
        (personBox.bottom - personBox.top) * marginFactor * height
    ) * 0.5f

    var left = (centerX - halfSide).coerceIn(0f, width).toInt()
    var top = (centerY - halfSide).coerceIn(0f, height).toInt()
    var right = (centerX + halfSide).coerceIn(0f, width).toInt()
    var bottom = (centerY + halfSide).coerceIn(0f, height).toInt()

    if (right <= left) {
        if (left >= source.width) {
            left = source.width - 1
        }
        right = (left + 1).coerceAtMost(source.width)
    }
    if (bottom <= top) {
        if (top >= source.height) {
            top = source.height - 1
        }
        bottom = (top + 1).coerceAtMost(source.height)
    }

    val cropped = Bitmap.createBitmap(source, left, top, right - left, bottom - top)
    val scaled = Bitmap.createScaledBitmap(
        cropped,
        PoseTensorContract.INPUT_SIZE,
        PoseTensorContract.INPUT_SIZE,
        true
    )
    if (cropped !== scaled) {
        cropped.recycle()
    }
    val cropRectNormalized = CropRectNormalized(
        left = left.toFloat() / source.width.toFloat(),
        top = top.toFloat() / source.height.toFloat(),
        right = right.toFloat() / source.width.toFloat(),
        bottom = bottom.toFloat() / source.height.toFloat()
    )
    return PersonCrop(
        bitmap = scaled,
        cropRectNormalized = cropRectNormalized
    )
}
