package com.courtvision.spike.pipeline

import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class KalmanBallTrackerTest {

    @Test
    fun predictOnly_extrapolatesPositionAfterVelocityLearned() {
        val tracker = KalmanBallTracker(maxMissFrames = 10)
        val first = tracker.track(dtSec = 0.1f, measurement = ballBoxAt(0.20f, 0.50f))
        val second = tracker.track(dtSec = 0.1f, measurement = ballBoxAt(0.30f, 0.50f))
        val predicted = tracker.track(dtSec = 0.1f, measurement = null)

        assertTrue(first.isTracked)
        assertTrue(second.isTracked)
        assertTrue(predicted.isTracked)
        assertTrue(predicted.centroidX > second.centroidX)
    }

    @Test
    fun update_reducesPositionCovarianceAfterPredict() {
        val tracker = KalmanBallTracker(maxMissFrames = 10)
        tracker.track(dtSec = 0.033f, measurement = ballBoxAt(0.4f, 0.5f))
        tracker.track(dtSec = 0.033f, measurement = null)
        val beforeUpdate = tracker.covarianceDiagonalForTest()

        tracker.track(dtSec = 0.033f, measurement = ballBoxAt(0.41f, 0.5f))
        val afterUpdate = tracker.covarianceDiagonalForTest()

        assertTrue(afterUpdate[0] < beforeUpdate[0])
        assertTrue(afterUpdate[1] < beforeUpdate[1])
    }

    @Test
    fun missFrames_resetExactlyAtThreshold() {
        val tracker = KalmanBallTracker(maxMissFrames = 2)
        tracker.track(dtSec = 0.033f, measurement = ballBoxAt(0.5f, 0.5f))

        val missOne = tracker.track(dtSec = 0.033f, measurement = null)
        val missTwo = tracker.track(dtSec = 0.033f, measurement = null)

        assertTrue(missOne.isTracked)
        assertFalse(missTwo.isTracked)
        assertTrue(tracker.missFrames == 0)
    }

    @Test
    fun stepChange_isSmoothedInsteadOfImmediateJump() {
        val tracker = KalmanBallTracker(maxMissFrames = 10)
        val start = tracker.track(dtSec = 0.033f, measurement = ballBoxAt(0.2f, 0.5f))
        val afterJumpMeasurement = tracker.track(dtSec = 0.033f, measurement = ballBoxAt(0.8f, 0.5f))

        assertTrue(start.isTracked)
        assertTrue(afterJumpMeasurement.isTracked)
        assertTrue(afterJumpMeasurement.centroidX > 0.2f)
        assertTrue(afterJumpMeasurement.centroidX < 0.8f)
    }

    @Test
    fun lowFps_10hz_staysStableWithSmallNoise() {
        val tracker = KalmanBallTracker(maxMissFrames = 10)
        val baseY = 0.55f
        var lastX = Float.NEGATIVE_INFINITY

        repeat(12) { index ->
            val baseX = 0.20f + index * 0.03f
            val noisyX = baseX + if (index % 2 == 0) 0.003f else -0.003f
            val noisyY = baseY + if (index % 3 == 0) 0.002f else -0.002f
            val tracked = tracker.track(
                dtSec = 0.1f,
                measurement = ballBoxAt(noisyX, noisyY)
            )
            assertTrue(tracked.isTracked)
            assertTrue(tracker.missFrames == 0)
            assertTrue(tracked.centroidX >= lastX - 0.01f)
            lastX = tracked.centroidX
        }
    }

    @Test
    fun lowFps_jitteredDt_maintainsBoundedVelocity() {
        val tracker = KalmanBallTracker(maxMissFrames = 10)

        repeat(14) { index ->
            val dt = if (index % 2 == 0) 0.07f else 0.13f
            val cx = 0.25f + index * 0.02f
            val tracked = tracker.track(
                dtSec = dt,
                measurement = ballBoxAt(cx, 0.52f)
            )

            assertTrue(tracked.isTracked)
            assertTrue(abs(tracked.velocityX) < 3.0f)
            assertTrue(abs(tracked.velocityY) < 3.0f)
        }
    }

    @Test
    fun dtClamp_handlesInvalidAndLargeIntervals() {
        val tracker = KalmanBallTracker(maxMissFrames = 10)
        tracker.track(dtSec = 0.1f, measurement = ballBoxAt(0.30f, 0.5f))

        val withZeroDt = tracker.track(dtSec = 0f, measurement = ballBoxAt(0.35f, 0.5f))
        val withNaNDt = tracker.track(dtSec = Float.NaN, measurement = ballBoxAt(0.40f, 0.5f))
        val withLargeDt = tracker.track(dtSec = 2.0f, measurement = ballBoxAt(0.45f, 0.5f))

        assertTrue(withZeroDt.isTracked)
        assertTrue(withNaNDt.isTracked)
        assertTrue(withLargeDt.isTracked)
        assertTrue(withLargeDt.centroidX in 0f..1f)
        assertTrue(withLargeDt.centroidY in 0f..1f)
    }

    @Test
    fun dropoutAt10fps_resetsAtConfiguredMissFrames() {
        assertResetsAtThreshold(maxMissFrames = 3)
        assertResetsAtThreshold(maxMissFrames = 10)
    }

    @Test
    fun reacquireAfterReset_reinitializesCleanly() {
        val tracker = KalmanBallTracker(maxMissFrames = 2)
        tracker.track(dtSec = 0.1f, measurement = ballBoxAt(0.30f, 0.5f))
        tracker.track(dtSec = 0.1f, measurement = ballBoxAt(0.40f, 0.5f))

        tracker.track(dtSec = 0.1f, measurement = null)
        val resetState = tracker.track(dtSec = 0.1f, measurement = null)
        assertFalse(resetState.isTracked)

        val reacquired = tracker.track(dtSec = 0.1f, measurement = ballBoxAt(0.25f, 0.45f))
        assertTrue(reacquired.isTracked)
        assertTrue(abs(reacquired.velocityX) < 1e-6f)
        assertTrue(abs(reacquired.velocityY) < 1e-6f)
        assertTrue(tracker.missFrames == 0)
    }

    @Test
    fun covariance_josephForm_staysFiniteNearSymmetricAndNonNegativeOnDiagonal() {
        val tracker = KalmanBallTracker(maxMissFrames = 10)

        repeat(60) { index ->
            val dt = when (index % 4) {
                0 -> 0.07f
                1 -> 0.13f
                2 -> 0.1f
                else -> 0.09f
            }
            val measurement = if (index % 7 == 0) {
                null
            } else {
                val cx = 0.25f + index * 0.008f + if (index % 2 == 0) 0.004f else -0.004f
                val cy = 0.55f + if (index % 3 == 0) 0.003f else -0.003f
                ballBoxAt(cx.coerceIn(0.05f, 0.95f), cy.coerceIn(0.05f, 0.95f))
            }
            tracker.track(dtSec = dt, measurement = measurement)
        }

        val covariance = tracker.covarianceMatrixForTest()
        for (i in covariance.indices) {
            assertTrue(covariance[i].isFinite())
        }
        for (diag in 0 until 4) {
            assertTrue(covariance[idx(diag, diag)] >= -1e-6f)
        }

        assertEquals(covariance[idx(0, 1)], covariance[idx(1, 0)], 1e-3f)
        assertEquals(covariance[idx(0, 2)], covariance[idx(2, 0)], 1e-3f)
        assertEquals(covariance[idx(0, 3)], covariance[idx(3, 0)], 1e-3f)
        assertEquals(covariance[idx(1, 2)], covariance[idx(2, 1)], 1e-3f)
        assertEquals(covariance[idx(1, 3)], covariance[idx(3, 1)], 1e-3f)
        assertEquals(covariance[idx(2, 3)], covariance[idx(3, 2)], 1e-3f)
    }

    private fun assertResetsAtThreshold(maxMissFrames: Int) {
        val tracker = KalmanBallTracker(maxMissFrames = maxMissFrames)
        tracker.track(dtSec = 0.1f, measurement = ballBoxAt(0.45f, 0.5f))
        repeat(maxMissFrames - 1) {
            val tracked = tracker.track(dtSec = 0.1f, measurement = null)
            assertTrue(tracked.isTracked)
        }

        val reset = tracker.track(dtSec = 0.1f, measurement = null)
        assertFalse(reset.isTracked)
        assertTrue(tracker.missFrames == 0)
    }

    private fun ballBoxAt(cx: Float, cy: Float): DetectionBox {
        val halfSize = 0.05f
        return DetectionBox(
            classId = 0,
            label = "ball",
            confidence = 0.95f,
            left = (cx - halfSize).coerceIn(0f, 1f),
            top = (cy - halfSize).coerceIn(0f, 1f),
            right = (cx + halfSize).coerceIn(0f, 1f),
            bottom = (cy + halfSize).coerceIn(0f, 1f)
        )
    }

    private fun idx(row: Int, col: Int): Int = row * 4 + col
}
