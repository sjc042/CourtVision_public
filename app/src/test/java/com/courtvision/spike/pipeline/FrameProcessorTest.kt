package com.courtvision.spike.pipeline

import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class FrameProcessorTest {

    @Test
    fun submitFrame_overflow_counts_once_per_evicted_frame() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)

        try {
            processor.submitFrame(frame(1))
            processor.submitFrame(frame(2))
            processor.submitFrame(frame(3))

            processor.publishWindowStatsForTest()

            assertEquals(2L, processor.stats.value.droppedFrames)
        } finally {
            processor.shutdown()
        }
    }

    @Test
    fun submitFrame_success_sets_queueDepth_to_one() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val processor = FrameProcessor(scope = this, consumerDispatcher = dispatcher)

        try {
            processor.submitFrame(frame(1))

            processor.publishWindowStatsForTest()

            assertEquals(1, processor.stats.value.queueDepth)
        } finally {
            processor.shutdown()
        }
    }

    private fun frame(id: Int) = FramePacket(
        timestampNs = id.toLong(),
        width = 1280,
        height = 720,
        rotationDegrees = 0,
        format = 35
    )
}

private fun FrameProcessor.publishWindowStatsForTest() {
    val method = FrameProcessor::class.java.getDeclaredMethod("publishWindowStats")
    method.isAccessible = true
    method.invoke(this)
}
