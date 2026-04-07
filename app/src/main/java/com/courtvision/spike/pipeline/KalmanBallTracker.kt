package com.courtvision.spike.pipeline

class KalmanBallTracker(
    processNoise: Float = 1e-2f,      // tuned 2026-04-07 from on-device CSVs
    measurementNoise: Float = 1e-5f,
    maxMissFrames: Int = DEFAULT_MAX_MISS_FRAMES
) {
    var processNoise: Float = processNoise
        set(value) {
            field = value.coerceAtLeast(1e-9f)
        }

    var measurementNoise: Float = measurementNoise
        set(value) {
            field = value.coerceAtLeast(1e-9f)
        }

    var maxMissFrames: Int = maxMissFrames.coerceAtLeast(1)
        set(value) {
            field = value.coerceAtLeast(1)
        }

    val missFrames: Int
        get() = missCount

    private val state = FloatArray(STATE_SIZE)
    private val covariance = FloatArray(MAT4_SIZE)
    private var initialized = false
    private var missCount = 0

    private val transition = FloatArray(MAT4_SIZE)
    private val transitionTranspose = FloatArray(MAT4_SIZE)
    private val processCovariance = FloatArray(MAT4_SIZE)
    private val tmp4x4A = FloatArray(MAT4_SIZE)
    private val tmp4x4B = FloatArray(MAT4_SIZE)
    private val kalmanGain = FloatArray(MAT4X2_SIZE)
    private val iMinusKh = FloatArray(MAT4_SIZE)
    private val iMinusKhTranspose = FloatArray(MAT4_SIZE)
    private val krkt = FloatArray(MAT4_SIZE)

    fun track(
        dtSec: Float,
        measurement: DetectionBox?
    ): TrackedBall {
        val safeDt = sanitizeDt(dtSec)
        if (measurement == null) {
            if (!initialized) {
                return TrackedBall(
                    centroidX = 0f,
                    centroidY = 0f,
                    velocityX = 0f,
                    velocityY = 0f,
                    isTracked = false,
                    rawBox = null
                )
            }

            predictInternal(safeDt)
            missCount += 1
            if (missCount >= maxMissFrames) {
                reset()
                return TrackedBall(
                    centroidX = 0f,
                    centroidY = 0f,
                    velocityX = 0f,
                    velocityY = 0f,
                    isTracked = false,
                    rawBox = null
                )
            }

            return currentTrackedBall(rawBox = null)
        }

        val centroidX = (measurement.left + measurement.right) * 0.5f
        val centroidY = (measurement.top + measurement.bottom) * 0.5f

        if (!initialized) {
            initialize(centroidX, centroidY)
        } else {
            predictInternal(safeDt)
            updateInternal(centroidX, centroidY)
        }
        missCount = 0
        return currentTrackedBall(rawBox = measurement)
    }

    fun reset() {
        initialized = false
        missCount = 0
        state.fill(0f)
        covariance.fill(0f)
    }

    internal fun covarianceDiagonalForTest(): FloatArray {
        return floatArrayOf(
            covariance[m4(0, 0)],
            covariance[m4(1, 1)],
            covariance[m4(2, 2)],
            covariance[m4(3, 3)]
        )
    }

    internal fun covarianceMatrixForTest(): FloatArray = covariance.copyOf()

    private fun initialize(centroidX: Float, centroidY: Float) {
        state[IDX_X] = centroidX.coerceIn(0f, 1f)
        state[IDX_Y] = centroidY.coerceIn(0f, 1f)
        state[IDX_VX] = 0f
        state[IDX_VY] = 0f
        covariance.fill(0f)
        covariance[m4(0, 0)] = INITIAL_POSITION_COVARIANCE
        covariance[m4(1, 1)] = INITIAL_POSITION_COVARIANCE
        covariance[m4(2, 2)] = INITIAL_VELOCITY_COVARIANCE
        covariance[m4(3, 3)] = INITIAL_VELOCITY_COVARIANCE
        initialized = true
    }

    private fun predictInternal(dtSec: Float) {
        buildTransition(dtSec)
        buildProcessCovariance(dtSec)

        val x = state[IDX_X]
        val y = state[IDX_Y]
        val vx = state[IDX_VX]
        val vy = state[IDX_VY]
        state[IDX_X] = (x + dtSec * vx).coerceIn(-POSITION_BOUNDARY, 1f + POSITION_BOUNDARY)
        state[IDX_Y] = (y + dtSec * vy).coerceIn(-POSITION_BOUNDARY, 1f + POSITION_BOUNDARY)

        mat4x4Mul(transition, covariance, tmp4x4A)
        mat4x4Mul(tmp4x4A, transitionTranspose, tmp4x4B)
        mat4x4Add(tmp4x4B, processCovariance, covariance)
    }

    private fun updateInternal(measX: Float, measY: Float) {
        val residualX = measX - state[IDX_X]
        val residualY = measY - state[IDX_Y]

        val s00 = covariance[m4(0, 0)] + measurementNoise
        val s01 = covariance[m4(0, 1)]
        val s10 = covariance[m4(1, 0)]
        val s11 = covariance[m4(1, 1)] + measurementNoise

        val det = (s00 * s11) - (s01 * s10)
        if (kotlin.math.abs(det) < SINGULARITY_EPSILON) {
            return
        }
        val invDet = 1f / det
        val sInv00 = s11 * invDet
        val sInv01 = -s01 * invDet
        val sInv10 = -s10 * invDet
        val sInv11 = s00 * invDet

        for (row in 0 until STATE_SIZE) {
            val p0 = covariance[m4(row, 0)]
            val p1 = covariance[m4(row, 1)]
            kalmanGain[m42(row, 0)] = p0 * sInv00 + p1 * sInv10
            kalmanGain[m42(row, 1)] = p0 * sInv01 + p1 * sInv11
        }

        for (row in 0 until STATE_SIZE) {
            val kx = kalmanGain[m42(row, 0)]
            val ky = kalmanGain[m42(row, 1)]
            state[row] += (kx * residualX) + (ky * residualY)
        }
        state[IDX_X] = state[IDX_X].coerceIn(-POSITION_BOUNDARY, 1f + POSITION_BOUNDARY)
        state[IDX_Y] = state[IDX_Y].coerceIn(-POSITION_BOUNDARY, 1f + POSITION_BOUNDARY)

        iMinusKh.fill(0f)
        iMinusKh[m4(0, 0)] = 1f - kalmanGain[m42(0, 0)]
        iMinusKh[m4(0, 1)] = -kalmanGain[m42(0, 1)]
        iMinusKh[m4(1, 0)] = -kalmanGain[m42(1, 0)]
        iMinusKh[m4(1, 1)] = 1f - kalmanGain[m42(1, 1)]
        iMinusKh[m4(2, 0)] = -kalmanGain[m42(2, 0)]
        iMinusKh[m4(2, 1)] = -kalmanGain[m42(2, 1)]
        iMinusKh[m4(3, 0)] = -kalmanGain[m42(3, 0)]
        iMinusKh[m4(3, 1)] = -kalmanGain[m42(3, 1)]
        iMinusKh[m4(2, 2)] = 1f
        iMinusKh[m4(3, 3)] = 1f

        // Joseph-form covariance update:
        // P = (I - K H) P (I - K H)^T + K R K^T
        mat4x4Mul(iMinusKh, covariance, tmp4x4A)
        transpose4x4(iMinusKh, iMinusKhTranspose)
        mat4x4Mul(tmp4x4A, iMinusKhTranspose, tmp4x4B)
        buildKrkt(measurementNoise)
        mat4x4Add(tmp4x4B, krkt, covariance)
    }

    private fun currentTrackedBall(rawBox: DetectionBox?): TrackedBall {
        return TrackedBall(
            centroidX = state[IDX_X].coerceIn(0f, 1f),
            centroidY = state[IDX_Y].coerceIn(0f, 1f),
            velocityX = state[IDX_VX],
            velocityY = state[IDX_VY],
            isTracked = initialized,
            rawBox = rawBox
        )
    }

    private fun buildTransition(dtSec: Float) {
        transition.fill(0f)
        transition[m4(0, 0)] = 1f
        transition[m4(0, 2)] = dtSec
        transition[m4(1, 1)] = 1f
        transition[m4(1, 3)] = dtSec
        transition[m4(2, 2)] = 1f
        transition[m4(3, 3)] = 1f
        transpose4x4(transition, transitionTranspose)
    }

    private fun buildProcessCovariance(dtSec: Float) {
        processCovariance.fill(0f)
        val posNoise = processNoise * dtSec * dtSec
        processCovariance[m4(0, 0)] = posNoise
        processCovariance[m4(1, 1)] = posNoise
        processCovariance[m4(2, 2)] = processNoise
        processCovariance[m4(3, 3)] = processNoise
    }

    private fun buildKrkt(measurementNoiseScalar: Float) {
        for (row in 0 until STATE_SIZE) {
            val kRow0 = kalmanGain[m42(row, 0)]
            val kRow1 = kalmanGain[m42(row, 1)]
            for (col in 0 until STATE_SIZE) {
                val kCol0 = kalmanGain[m42(col, 0)]
                val kCol1 = kalmanGain[m42(col, 1)]
                krkt[m4(row, col)] = measurementNoiseScalar * ((kRow0 * kCol0) + (kRow1 * kCol1))
            }
        }
    }

    private fun sanitizeDt(dtSec: Float): Float {
        if (!dtSec.isFinite()) return DEFAULT_DT_SEC
        if (dtSec <= 0f) return DEFAULT_DT_SEC
        return dtSec.coerceIn(MIN_DT_SEC, MAX_DT_SEC)
    }

    companion object {
        private const val STATE_SIZE = 4
        private const val MAT4_SIZE = 16
        private const val MAT4X2_SIZE = 8

        private const val IDX_X = 0
        private const val IDX_Y = 1
        private const val IDX_VX = 2
        private const val IDX_VY = 3

        private const val DEFAULT_DT_SEC = 1f / 30f
        private const val MIN_DT_SEC = 1f / 120f
        private const val MAX_DT_SEC = 0.25f
        private const val POSITION_BOUNDARY = 0.5f
        private const val INITIAL_POSITION_COVARIANCE = 0.1f
        private const val INITIAL_VELOCITY_COVARIANCE = 1f
        private const val SINGULARITY_EPSILON = 1e-9f
        private const val DEFAULT_MAX_MISS_FRAMES = 10

        private fun m4(row: Int, col: Int): Int = row * 4 + col
        private fun m42(row: Int, col: Int): Int = row * 2 + col
    }
}

private fun mat4x4Mul(left: FloatArray, right: FloatArray, out: FloatArray) {
    for (row in 0 until 4) {
        for (col in 0 until 4) {
            var sum = 0f
            for (k in 0 until 4) {
                sum += left[row * 4 + k] * right[k * 4 + col]
            }
            out[row * 4 + col] = sum
        }
    }
}

private fun mat4x4Add(left: FloatArray, right: FloatArray, out: FloatArray) {
    for (i in 0 until 16) {
        out[i] = left[i] + right[i]
    }
}

private fun transpose4x4(input: FloatArray, out: FloatArray) {
    for (row in 0 until 4) {
        for (col in 0 until 4) {
            out[col * 4 + row] = input[row * 4 + col]
        }
    }
}
