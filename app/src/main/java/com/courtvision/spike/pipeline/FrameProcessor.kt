package com.courtvision.spike.pipeline

import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageProxy
import java.nio.MappedByteBuffer
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op

enum class ModelOutputFormat { RAW_8400, END_TO_END_300 }

class FrameProcessor(
    private val scope: CoroutineScope,
    private val modelBufferProvider: (() -> MappedByteBuffer)? = null,
    private val consumerDispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1),
    private val thermalStatusProvider: () -> String = { "UNKNOWN" }
) : FrameConsumer, FrameProcessorGateway {

    private val droppedByOverflow = AtomicLong(0)
    private val queueDepth = AtomicInteger(0)
    private val lastInferenceMs = AtomicReference(0.0)
    private val windowDurationsMs = ConcurrentLinkedQueue<Double>()
    private val pendingMode = AtomicReference<InferenceMode?>(null)
    private val currentMode = AtomicReference(InferenceMode.CPU)

    private val _stats = MutableStateFlow(PipelineStats())
    override val stats: StateFlow<PipelineStats> = _stats.asStateFlow()

    private val _detections = MutableStateFlow(DetectionFrame())
    override val detections: StateFlow<DetectionFrame> = _detections.asStateFlow()

    private var lastLoggedRotation = -1

    private val _isSwitchingMode = MutableStateFlow(false)
    override val isSwitchingMode: StateFlow<Boolean> = _isSwitchingMode.asStateFlow()

    private val _lastError = MutableStateFlow<String?>(null)
    override val lastError: StateFlow<String?> = _lastError.asStateFlow()

    // Rot90Op takes counter-clockwise 90° rotation count.
    // CameraX rotationDegrees is clockwise, so:
    //   0°   → 0 rotations
    //   90°  → 3 counter-clockwise (= 1 clockwise)
    //   180° → 2
    //   270° → 1 counter-clockwise (= 3 clockwise)
    private val imageProcessors: Map<Int, ImageProcessor> = mapOf(
        0   to buildImageProcessor(rot90count = 0),
        90  to buildImageProcessor(rot90count = 3),
        180 to buildImageProcessor(rot90count = 2),
        270 to buildImageProcessor(rot90count = 1),
    )

    private fun buildImageProcessor(rot90count: Int): ImageProcessor {
        return ImageProcessor.Builder().apply {
            if (rot90count > 0) add(Rot90Op(rot90count))
            add(ResizeOp(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            add(NormalizeOp(0f, 255f))
        }.build()
    }

    private lateinit var outputTensorRaw: Array<Array<FloatArray>>
    private val outputTensorE2E = Array(1) { Array(E2E_MAX_DETS) { FloatArray(E2E_FIELDS) } }
    private var outputFormat = ModelOutputFormat.RAW_8400

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    private val frameChannel = Channel<FrameTask>(
        capacity = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
        onUndeliveredElement = { task ->
            if (task is FrameTask.Image) {
                task.image.close()
            }
            droppedByOverflow.incrementAndGet()
        }
    )

    private val consumerJob: Job
    private val aggregateJob: Job

    init {
        consumerJob = scope.launch(consumerDispatcher) {
            for (task in frameChannel) {
                queueDepth.set(0)
                maybeApplyPendingMode()
                val startNs = System.nanoTime()
                when (task) {
                    is FrameTask.Metadata -> consume(task.frame)
                    is FrameTask.Image -> processImage(task.image)
                }
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
        val result = frameChannel.trySend(FrameTask.Metadata(frame))
        if (result.isSuccess) {
            queueDepth.set(1)
        }
    }

    override fun submitImage(image: ImageProxy) {
        if (frameChannel.isClosedForSend) {
            image.close()
            return
        }

        val result = frameChannel.trySend(FrameTask.Image(image))
        if (result.isSuccess) {
            queueDepth.set(1)
        } else {
            image.close()
        }
    }

    override fun setInferenceMode(mode: InferenceMode) {
        if (mode == currentMode.get() && pendingMode.get() == null) {
            _isSwitchingMode.value = false
            return
        }
        pendingMode.set(mode)
        _isSwitchingMode.value = true
        scope.launch(consumerDispatcher) {
            maybeApplyPendingMode(force = true)
        }
    }

    override suspend fun consume(frame: FramePacket) {
        // Metadata-only path used by Day 1-2 tests to validate queue/drop behavior.
    }

    override fun resetInterpreter() {
        scope.launch(consumerDispatcher) {
            closeInterpreterResources()
        }
    }

    override fun shutdown() {
        frameChannel.close()
        closeInterpreterResources()
        consumerJob.cancel()
        aggregateJob.cancel()
    }

    private fun maybeApplyPendingMode(force: Boolean = false) {
        val requestedMode = pendingMode.getAndSet(null)
        if (requestedMode == null && !force) return

        val target = requestedMode ?: currentMode.get()
        if (target == currentMode.get() && interpreter != null) {
            _isSwitchingMode.value = false
            return
        }

        val switched = switchInterpreter(target)
        if (!switched && target == InferenceMode.GPU) {
            switchInterpreter(InferenceMode.CPU)
        }
        _isSwitchingMode.value = false
    }

    private fun switchInterpreter(mode: InferenceMode): Boolean {
        closeInterpreterResources()

        val mappedModel = modelBufferProvider?.invoke()
        if (mappedModel == null) {
            _lastError.value = "Model buffer provider unavailable"
            currentMode.set(InferenceMode.CPU)
            return false
        }

        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }

        val localDelegate = if (mode == InferenceMode.GPU) {
            val compatibility = CompatibilityList()
            if (!compatibility.isDelegateSupportedOnThisDevice) {
                _lastError.value = "GPU delegate unsupported on this device"
                return false
            }
            GpuDelegate(compatibility.bestOptionsForThisDevice).also {
                options.addDelegate(it)
            }
        } else {
            null
        }

        return try {
            val localInterpreter = Interpreter(mappedModel, options)
            validateTensorContract(localInterpreter)
            interpreter = localInterpreter
            gpuDelegate = localDelegate
            currentMode.set(mode)
            _lastError.value = null
            true
        } catch (error: Throwable) {
            localDelegate?.close()
            _lastError.value = "Interpreter init failed: ${error.message ?: "unknown error"}"
            false
        }
    }

    private fun validateTensorContract(localInterpreter: Interpreter) {
        val inputShape = localInterpreter.getInputTensor(0).shape()
        val outputShape = localInterpreter.getOutputTensor(0).shape()
        val inputType = localInterpreter.getInputTensor(0).dataType()
        val outputType = localInterpreter.getOutputTensor(0).dataType()

        require(inputShape.contentEquals(intArrayOf(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3))) {
            "Unexpected input shape: ${inputShape.contentToString()}"
        }

        val isRaw = outputShape.size == 3 &&
                    outputShape[0] == 1 &&
                    outputShape[1] >= 5 &&
                    outputShape[2] == OUTPUT_BOXES
        val isE2E = outputShape.contentEquals(intArrayOf(1, E2E_MAX_DETS, E2E_FIELDS))
        require(isRaw || isE2E) {
            "Unexpected output shape: ${outputShape.contentToString()}"
        }
        outputFormat = if (isE2E) ModelOutputFormat.END_TO_END_300 else ModelOutputFormat.RAW_8400
        if (isRaw) {
            outputTensorRaw = Array(1) { Array(outputShape[1]) { FloatArray(OUTPUT_BOXES) } }
        }

        require(inputType == DataType.FLOAT32) {
            "Unexpected input dtype: $inputType"
        }
        require(outputType == DataType.FLOAT32) {
            "Unexpected output dtype: $outputType"
        }
    }

    private fun closeInterpreterResources() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
    }

    private suspend fun processImage(image: ImageProxy) {
        try {
            if (interpreter == null) {
                maybeApplyPendingMode(force = true)
            }
            val localInterpreter = interpreter ?: return

            val rotationDegrees = image.imageInfo.rotationDegrees
            val normalizedRotation = ((rotationDegrees % 360) + 360) % 360
            val rotationChanged = normalizedRotation != lastLoggedRotation

            val bitmap = image.toBitmap()
            val processor = imageProcessors[normalizedRotation] ?: imageProcessors[0]!!
            val rot90count = when (normalizedRotation) { 90 -> 3; 180 -> 2; 270 -> 1; else -> 0 }
            val tensorImage = processor.process(TensorImage.fromBitmap(bitmap))

            val bitmapW = bitmap.width
            val bitmapH = bitmap.height
            val rotatedWidth: Int
            val rotatedHeight: Int
            if (normalizedRotation == 90 || normalizedRotation == 270) {
                rotatedWidth = bitmapH
                rotatedHeight = bitmapW
            } else {
                rotatedWidth = bitmapW
                rotatedHeight = bitmapH
            }
            bitmap.recycle()

            if (rotationChanged) {
                Log.d("CV_Rotation", "ROTATION raw=$rotationDegrees normalized=$normalizedRotation")
                Log.d("CV_Rotation", "PREPROCESS bitmapW=$bitmapW bitmapH=$bitmapH tensorW=${tensorImage.width} tensorH=${tensorImage.height} rot90count=$rot90count")
                Log.d("CV_Rotation", "DIMS rotatedW=$rotatedWidth rotatedH=$rotatedHeight")
                lastLoggedRotation = normalizedRotation
            }

            val inputBuffer = tensorImage.buffer
            val inferenceStartNs = System.nanoTime()
            val boxes = when (outputFormat) {
                ModelOutputFormat.RAW_8400 -> {
                    for (row in outputTensorRaw[0]) row.fill(0f)
                    localInterpreter.run(inputBuffer, outputTensorRaw)
                    parseModelOutput(outputTensorRaw, CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD)
                }
                ModelOutputFormat.END_TO_END_300 -> {
                    for (row in outputTensorE2E[0]) row.fill(0f)
                    localInterpreter.run(inputBuffer, outputTensorE2E)
                    parseEndToEndOutput(outputTensorE2E, CONFIDENCE_THRESHOLD)
                }
            }
            val inferenceMs = (System.nanoTime() - inferenceStartNs) / 1_000_000.0
            lastInferenceMs.set(inferenceMs)
            if (rotationChanged && boxes.isNotEmpty()) {
                val b = boxes[0]
                Log.d("CV_Rotation", "BOX[0] class=${b.label} conf=${"%.2f".format(b.confidence)} ltrb=[${b.left}, ${b.top}, ${b.right}, ${b.bottom}]")
            }
            if (rotationChanged) {
                Log.d("CV_Rotation", "EMIT rotation=0(hardcoded) srcW=$rotatedWidth srcH=$rotatedHeight boxCount=${boxes.size}")
            }
            _detections.value = DetectionFrame(
                timestampNs = image.imageInfo.timestamp,
                sourceWidth = rotatedWidth,
                sourceHeight = rotatedHeight,
                rotationDegrees = 0,
                boxes = boxes
            )
        } catch (error: Throwable) {
            _lastError.value = "Frame processing failed: ${error.message ?: "unknown error"}"
        } finally {
            image.close()
        }
    }

    internal fun parseModelOutput(
        output: Array<Array<FloatArray>>,
        confidenceThreshold: Float,
        iouThreshold: Float
    ): List<DetectionBox> {
        val perClassCandidates = mutableMapOf<Int, MutableList<DetectionBox>>()

        val numClasses = output[0].size - 4
        for (candidateIndex in 0 until OUTPUT_BOXES) {
            val cx = output[0][0][candidateIndex]
            val cy = output[0][1][candidateIndex]
            val w = output[0][2][candidateIndex]
            val h = output[0][3][candidateIndex]

            var bestClassId = -1
            var bestScore = 0f
            for (classIndex in 0 until numClasses) {
                val score = output[0][4 + classIndex][candidateIndex]
                if (score > bestScore) {
                    bestScore = score
                    bestClassId = classIndex
                }
            }

            if (bestScore < confidenceThreshold || bestClassId < 0 || bestClassId >= CUSTOM_CLASS_NAMES.size) continue

            val left = (cx - w / 2f).coerceIn(0f, 1f)
            val top = (cy - h / 2f).coerceIn(0f, 1f)
            val right = (cx + w / 2f).coerceIn(0f, 1f)
            val bottom = (cy + h / 2f).coerceIn(0f, 1f)

            if (right <= left || bottom <= top) continue

            perClassCandidates
                .getOrPut(bestClassId) { mutableListOf() }
                .add(
                    DetectionBox(
                        classId = bestClassId,
                        label = classLabel(bestClassId),
                        confidence = bestScore,
                        left = left,
                        top = top,
                        right = right,
                        bottom = bottom
                    )
                )
        }

        return perClassCandidates
            .values
            .asSequence()
            .flatMap { nms(it, iouThreshold).asSequence() }
            .sortedByDescending { it.confidence }
            .toList()
    }

    internal fun parseEndToEndOutput(
        output: Array<Array<FloatArray>>,
        confidenceThreshold: Float
    ): List<DetectionBox> {
        val results = mutableListOf<DetectionBox>()
        for (i in 0 until E2E_MAX_DETS) {
            val x1 = output[0][i][0]
            val y1 = output[0][i][1]
            val x2 = output[0][i][2]
            val y2 = output[0][i][3]
            val conf = output[0][i][4]
            val classId = output[0][i][5].toInt()

            if (conf < confidenceThreshold || classId < 0 || classId >= CUSTOM_CLASS_NAMES.size) continue

            val left = x1.coerceIn(0f, 1f)
            val top = y1.coerceIn(0f, 1f)
            val right = x2.coerceIn(0f, 1f)
            val bottom = y2.coerceIn(0f, 1f)

            if (right <= left || bottom <= top) continue

            results.add(
                DetectionBox(
                    classId = classId,
                    label = classLabel(classId),
                    confidence = conf,
                    left = left,
                    top = top,
                    right = right,
                    bottom = bottom
                )
            )
        }
        return results.sortedByDescending { it.confidence }
    }

    private fun nms(boxes: List<DetectionBox>, iouThreshold: Float): List<DetectionBox> {
        if (boxes.isEmpty()) return emptyList()
        val remaining = boxes.sortedByDescending { it.confidence }.toMutableList()
        val selected = mutableListOf<DetectionBox>()

        while (remaining.isNotEmpty()) {
            val best = remaining.removeAt(0)
            selected.add(best)
            val iterator = remaining.iterator()
            while (iterator.hasNext()) {
                val next = iterator.next()
                if (iou(best, next) > iouThreshold) {
                    iterator.remove()
                }
            }
        }
        return selected
    }

    private fun iou(a: DetectionBox, b: DetectionBox): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)

        val interWidth = (interRight - interLeft).coerceAtLeast(0f)
        val interHeight = (interBottom - interTop).coerceAtLeast(0f)
        val interArea = interWidth * interHeight
        if (interArea <= 0f) return 0f

        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        val union = areaA + areaB - interArea
        return if (union <= 0f) 0f else interArea / union
    }

    private fun classLabel(classId: Int): String =
        CUSTOM_CLASS_NAMES.getOrElse(classId) { "class$classId" }

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
                queueDepth = queueDepth.get(),
                lastInferenceMs = lastInferenceMs.get(),
                delegateMode = currentMode.get(),
                ramMb = currentProcessRamMb(),
                thermalStatus = thermalStatusProvider()
            )
        }
    }

    private fun currentProcessRamMb(): Double {
        val runtime = Runtime.getRuntime()
        val usedBytes = runtime.totalMemory() - runtime.freeMemory()
        return usedBytes / (1024.0 * 1024.0)
    }

    private fun calculateP95(samples: List<Double>): Double {
        if (samples.isEmpty()) return 0.0
        val sorted = samples.sorted()
        val index = floor((sorted.size - 1) * 0.95).toInt()
        return sorted[index]
    }

    private sealed interface FrameTask {
        data class Metadata(val frame: FramePacket) : FrameTask
        data class Image(val image: ImageProxy) : FrameTask
    }

    companion object {
        private const val MODEL_INPUT_SIZE = 640
        private const val OUTPUT_BOXES = 8400
        private const val CONFIDENCE_THRESHOLD = 0.40f
        private const val NMS_IOU_THRESHOLD = 0.50f

        private const val E2E_MAX_DETS = 300
        private const val E2E_FIELDS = 6

        private val CUSTOM_CLASS_NAMES = arrayOf("ball", "made", "person", "rim", "shoot")
    }
}
