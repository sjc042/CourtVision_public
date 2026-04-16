package com.courtvision.spike.pipeline

import android.graphics.Bitmap
import android.graphics.Color
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
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
}
