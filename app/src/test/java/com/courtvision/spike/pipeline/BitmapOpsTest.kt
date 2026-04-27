package com.courtvision.spike.pipeline

import android.graphics.Bitmap
import android.graphics.Color
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertSame
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.extension.ExtendWith
import tech.apter.junit.jupiter.robolectric.RobolectricExtension

@ExtendWith(RobolectricExtension::class)
class BitmapOpsTest {

    @Test
    fun squarePadCrop_squareInput_returnsSameInstance() {
        val source = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        source.eraseColor(Color.WHITE)

        try {
            val padded = squarePadCrop(source)
            assertSame(source, padded)
        } finally {
            source.recycle()
        }
    }

    @Test
    fun squarePadCrop_tallInput_returnsSquareAndCenteredContent() {
        val source = Bitmap.createBitmap(100, 200, Bitmap.Config.ARGB_8888)
        source.eraseColor(Color.WHITE)

        val padded = squarePadCrop(source)
        try {
            assertEquals(200, padded.width)
            assertEquals(200, padded.height)

            // Left pad, center content, right pad.
            assertEquals(Color.BLACK, padded.getPixel(25, 100))
            assertEquals(Color.WHITE, padded.getPixel(75, 100))
            assertEquals(Color.BLACK, padded.getPixel(175, 100))
        } finally {
            if (padded !== source) padded.recycle()
            source.recycle()
        }
    }

    @Test
    fun squarePadCrop_wideInput_returnsSquareAndCenteredContent() {
        val source = Bitmap.createBitmap(300, 100, Bitmap.Config.ARGB_8888)
        source.eraseColor(Color.WHITE)

        val padded = squarePadCrop(source)
        try {
            assertEquals(300, padded.width)
            assertEquals(300, padded.height)

            // Top pad, center content, bottom pad.
            assertEquals(Color.BLACK, padded.getPixel(150, 50))
            assertEquals(Color.WHITE, padded.getPixel(150, 150))
            assertEquals(Color.BLACK, padded.getPixel(150, 275))
        } finally {
            if (padded !== source) padded.recycle()
            source.recycle()
        }
    }

    @Test
    fun squarePadCrop_withBbox_returns256x256() {
        val source = Bitmap.createBitmap(400, 600, Bitmap.Config.ARGB_8888)
        source.eraseColor(Color.WHITE)
        val personBox = DetectionBox(
            classId = 2,
            label = "person",
            confidence = 0.9f,
            left = 0.3f,
            top = 0.2f,
            right = 0.5f,
            bottom = 0.6f
        )

        val personCrop = squarePadCrop(source, personBox, marginFactor = 1.25f)
        try {
            assertEquals(PoseTensorContract.INPUT_SIZE, personCrop.bitmap.width)
            assertEquals(PoseTensorContract.INPUT_SIZE, personCrop.bitmap.height)
        } finally {
            personCrop.bitmap.recycle()
            source.recycle()
        }
    }

    @Test
    fun squarePadCrop_bboxAtEdge_clampsToFrameBoundsAndReturns256x256() {
        val source = Bitmap.createBitmap(300, 300, Bitmap.Config.ARGB_8888)
        source.eraseColor(Color.WHITE)
        val personBox = DetectionBox(
            classId = 2,
            label = "person",
            confidence = 0.8f,
            left = 0.85f,
            top = 0.85f,
            right = 1f,
            bottom = 1f
        )

        val personCrop = squarePadCrop(source, personBox)
        try {
            assertEquals(PoseTensorContract.INPUT_SIZE, personCrop.bitmap.width)
            assertEquals(PoseTensorContract.INPUT_SIZE, personCrop.bitmap.height)
        } finally {
            personCrop.bitmap.recycle()
            source.recycle()
        }
    }

    @Test
    fun squarePadCrop_bboxFullFrame_returns256x256() {
        val source = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
        source.eraseColor(Color.WHITE)
        val personBox = DetectionBox(
            classId = 2,
            label = "person",
            confidence = 0.95f,
            left = 0f,
            top = 0f,
            right = 1f,
            bottom = 1f
        )

        val personCrop = squarePadCrop(source, personBox, marginFactor = 1f)
        try {
            assertEquals(PoseTensorContract.INPUT_SIZE, personCrop.bitmap.width)
            assertEquals(PoseTensorContract.INPUT_SIZE, personCrop.bitmap.height)
        } finally {
            personCrop.bitmap.recycle()
            source.recycle()
        }
    }

    @Test
    fun squarePadCrop_bboxOverload_returnsClampedCropRect() {
        val source = Bitmap.createBitmap(400, 600, Bitmap.Config.ARGB_8888)
        source.eraseColor(Color.WHITE)
        val personBox = DetectionBox(
            classId = 2,
            label = "person",
            confidence = 0.9f,
            left = 0.3f,
            top = 0.2f,
            right = 0.5f,
            bottom = 0.6f
        )

        val personCrop = squarePadCrop(source, personBox, marginFactor = 1.25f)
        try {
            val rect = personCrop.cropRectNormalized
            val width = source.width.toFloat()
            val height = source.height.toFloat()
            val centerX = (personBox.left + personBox.right) * 0.5f * width
            val centerY = (personBox.top + personBox.bottom) * 0.5f * height
            val halfSide = maxOf(
                (personBox.right - personBox.left) * 1.25f * width,
                (personBox.bottom - personBox.top) * 1.25f * height
            ) * 0.5f

            var expectedLeft = (centerX - halfSide).coerceIn(0f, width).toInt()
            var expectedTop = (centerY - halfSide).coerceIn(0f, height).toInt()
            var expectedRight = (centerX + halfSide).coerceIn(0f, width).toInt()
            var expectedBottom = (centerY + halfSide).coerceIn(0f, height).toInt()

            if (expectedRight <= expectedLeft) {
                if (expectedLeft >= source.width) expectedLeft = source.width - 1
                expectedRight = (expectedLeft + 1).coerceAtMost(source.width)
            }
            if (expectedBottom <= expectedTop) {
                if (expectedTop >= source.height) expectedTop = source.height - 1
                expectedBottom = (expectedTop + 1).coerceAtMost(source.height)
            }

            assertEquals(expectedLeft.toFloat() / source.width, rect.left, 1f / source.width)
            assertEquals(expectedTop.toFloat() / source.height, rect.top, 1f / source.height)
            assertEquals(expectedRight.toFloat() / source.width, rect.right, 1f / source.width)
            assertEquals(expectedBottom.toFloat() / source.height, rect.bottom, 1f / source.height)
        } finally {
            personCrop.bitmap.recycle()
            source.recycle()
        }
    }

    @Test
    fun squarePadCrop_bboxAtEdge_clampsRectBounds() {
        val source = Bitmap.createBitmap(300, 300, Bitmap.Config.ARGB_8888)
        source.eraseColor(Color.WHITE)
        val personBox = DetectionBox(
            classId = 2,
            label = "person",
            confidence = 0.8f,
            left = 0.85f,
            top = 0.85f,
            right = 1f,
            bottom = 1f
        )

        val personCrop = squarePadCrop(source, personBox)
        try {
            val rect = personCrop.cropRectNormalized
            assertEquals(1f, rect.right, 0f)
            assertEquals(1f, rect.bottom, 0f)
            assertTrue(rect.left >= 0f)
            assertTrue(rect.top >= 0f)
        } finally {
            personCrop.bitmap.recycle()
            source.recycle()
        }
    }
}
