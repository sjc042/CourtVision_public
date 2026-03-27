package com.courtvision.spike.pipeline

import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.floor
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

class FrameProcessor(
    private val scope: CoroutineScope,
    private val consumerDispatcher: CoroutineDispatcher = Dispatchers.Default
) : FrameConsumer {

    private val droppedByOverflow = AtomicLong(0)
    private val queueDepth = AtomicInteger(0)
    private val simulatedDelayMs = AtomicLong(0)
    private val windowDurationsMs = ConcurrentLinkedQueue<Double>()

    private val _stats = MutableStateFlow(PipelineStats())
    val stats: StateFlow<PipelineStats> = _stats.asStateFlow()

    private val frameChannel = Channel<FramePacket>(
        capacity = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
        onUndeliveredElement = { _ ->
            droppedByOverflow.incrementAndGet()
        }
    )

    private val consumerJob: Job
    private val aggregateJob: Job

    init {
        consumerJob = scope.launch(consumerDispatcher) {
            for (frame in frameChannel) {
                queueDepth.set(0)
                val startNs = System.nanoTime()
                consume(frame)
                val elapsedMs = (System.nanoTime() - startNs) / 1_000_000.0
                windowDurationsMs.add(elapsedMs)
            }
        }

        aggregateJob = scope.launch {
            while (isActive) {
                delay(1_000)
                publishWindowStats()
            }
        }
    }

    fun submitFrame(frame: FramePacket) {
        if (frameChannel.isClosedForSend) return

        val result = frameChannel.trySend(frame)
        if (result.isSuccess) {
            queueDepth.set(1)
        }
    }

    fun setSimulatedDelayMs(delayMs: Long) {
        simulatedDelayMs.set(delayMs.coerceAtLeast(0))
    }

    override suspend fun consume(frame: FramePacket) {
        val delayMs = simulatedDelayMs.get()
        if (delayMs > 0) {
            delay(delayMs)
        }
    }

    fun shutdown() {
        frameChannel.close()
        consumerJob.cancel()
        aggregateJob.cancel()
    }

    private fun publishWindowStats() {
        val samples = mutableListOf<Double>()
        while (true) {
            val next = windowDurationsMs.poll() ?: break
            samples.add(next)
        }

        val fps = samples.size
        val avg = if (samples.isEmpty()) 0.0 else samples.average()
        val p95 = calculateP95(samples)

        _stats.update {
            it.copy(
                analysisFps = fps,
                avgAnalyzeMs = avg,
                p95AnalyzeMs = p95,
                droppedFrames = droppedByOverflow.get(),
                queueDepth = queueDepth.get()
            )
        }
    }

    private fun calculateP95(samples: List<Double>): Double {
        if (samples.isEmpty()) return 0.0
        val sorted = samples.sorted()
        val index = floor((sorted.size - 1) * 0.95).toInt()
        return sorted[index]
    }
}
