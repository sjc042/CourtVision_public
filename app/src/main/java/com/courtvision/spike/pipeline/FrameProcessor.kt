package com.courtvision.spike.pipeline

import android.graphics.Bitmap
import android.util.Log
import android.os.SystemClock
import androidx.annotation.VisibleForTesting
import androidx.camera.core.ImageProxy
import java.nio.MappedByteBuffer
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean
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
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

enum class ModelOutputFormat { RAW_8400, END_TO_END_300 }

class FrameProcessor(
    private val scope: CoroutineScope,
    private val modelBufferProvider: (() -> MappedByteBuffer)? = null,
    private val poseModelBufferProvider: (() -> MappedByteBuffer)? = null,
    private val poseInterpreterFactory: (MappedByteBuffer, Boolean) -> PoseInferenceEngine =
        { modelBuffer, useGpu -> PoseLandmarkInterpreter(modelBuffer, useGpu = useGpu) },
    private val consumerDispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1),
    private val thermalStatusProvider: () -> String = { "UNKNOWN" },
    private val perFrameLogger: PerFramePerfLogger? = null,
    private val perFrameMode: String = "sequential_yolo_pose",
    private val perFrameDevice: String = "UNKNOWN"
) : FrameConsumer, FrameProcessorGateway {

    private val droppedByOverflow = AtomicLong(0)
    private val queueDepth = AtomicInteger(0)
    private val lastInferenceMs = AtomicReference(0.0)
    private val windowDurationsMs = ConcurrentLinkedQueue<Double>()
    private val pendingMode = AtomicReference<InferenceMode?>(null)
    private val currentMode = AtomicReference(InferenceMode.CPU)
    private val poseGatingMode = AtomicReference(PoseGatingMode.EVERY_FRAME_WITH_PERSON)
    private val personSelectionMode = AtomicReference(PersonSelectionMode.HIGHEST_CONFIDENCE)
    private val lastFrameSkippedPose = AtomicBoolean(true)
    private val latestTrackedBall = AtomicReference<TrackedBall?>(null)
    private val latestMissStreak = AtomicInteger(0)
    private val expectedTargetRotation = AtomicInteger(-1)
    private val expectedFrameRotationDegrees = AtomicInteger(-1)
    private val ballTracker = KalmanBallTracker()
    private var lastNormalizedRotation: Int = -1
    private var lastImageTimestampNs: Long = -1L
    private var lastTrackerTimestampNs: Long = -1L
    private var rotationMismatchStartElapsedMs: Long = -1L
    private var lastRotationStallLogElapsedMs: Long = -1L
    private val perFrameTimingAccumulator = PerFrameTimingAccumulator()

    private val _stats = MutableStateFlow(PipelineStats())
    override val stats: StateFlow<PipelineStats> = _stats.asStateFlow()

    private val _detections = MutableStateFlow(DetectionFrame())
    override val detections: StateFlow<DetectionFrame> = _detections.asStateFlow()

    private val _poseResult = MutableStateFlow<LivePoseOverlay?>(null)
    override val poseResult: StateFlow<LivePoseOverlay?> = _poseResult.asStateFlow()

    private val _isSwitchingMode = MutableStateFlow(false)
    override val isSwitchingMode: StateFlow<Boolean> = _isSwitchingMode.asStateFlow()

    private val _lastError = MutableStateFlow<String?>(null)
    override val lastError: StateFlow<String?> = _lastError.asStateFlow()

    private val _rotationTelemetry = MutableStateFlow(RotationTelemetry())
    override val rotationTelemetry: StateFlow<RotationTelemetry> = _rotationTelemetry.asStateFlow()
    private val poseValidationQueue = ConcurrentLinkedQueue<PoseValidationEntry>()
    private val poseValidationActive = AtomicBoolean(false)
    private val poseValidationComplete = AtomicBoolean(false)
    private val poseValidationResultChannel = Channel<PoseFrameResult>(Channel.UNLIMITED)
    override val poseValidationResults: Flow<PoseFrameResult> = poseValidationResultChannel.receiveAsFlow()

    private val yoloImageProcessor: ImageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(0f, 255f))
        .build()
    private val yoloTensorImage = TensorImage(DataType.FLOAT32)

    private lateinit var outputTensorRaw: Array<Array<FloatArray>>
    private val outputTensorE2E = Array(1) { Array(E2E_MAX_DETS) { FloatArray(E2E_FIELDS) } }
    private var outputFormat = ModelOutputFormat.RAW_8400

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null
    private var poseInterpreter: PoseInferenceEngine? = null

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
        if (poseValidationComplete.get()) {
            image.close()
            return
        }
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

    override fun setPoseGatingMode(mode: PoseGatingMode) {
        poseGatingMode.set(mode)
    }

    override fun setPersonSelectionMode(mode: PersonSelectionMode) {
        personSelectionMode.set(mode)
    }

    override fun setTrackerMaxMissFrames(maxMissFrames: Int) {
        scope.launch(consumerDispatcher) {
            ballTracker.maxMissFrames = maxMissFrames
        }
    }

    override fun setTrackerNoise(processNoise: Float, measurementNoise: Float) {
        scope.launch(consumerDispatcher) {
            ballTracker.processNoise = processNoise
            ballTracker.measurementNoise = measurementNoise
        }
    }

    override fun updateExpectedRotation(
        expectedTargetRotation: Int,
        expectedFrameRotationDegrees: Int,
        source: String
    ) {
        val normalizedTarget = when (expectedTargetRotation) {
            0, 1, 2, 3 -> expectedTargetRotation
            else -> 0
        }
        val normalizedFrame = when (expectedFrameRotationDegrees) {
            0, 90, 180, 270 -> expectedFrameRotationDegrees
            else -> -1
        }

        this.expectedTargetRotation.set(normalizedTarget)
        this.expectedFrameRotationDegrees.set(normalizedFrame)
        _rotationTelemetry.update {
            it.copy(
                expectedTargetRotation = normalizedTarget,
                expectedFrameRotationDegrees = normalizedFrame
            )
        }
        if (ROTATION_DEBUG_LOGS) {
            Log.i(
                TAG,
                "[FRAME][EXPECTED_ROT] source=$source flag=EXPECTED_ROT expectedTarget=$normalizedTarget expectedRaw=$normalizedFrame"
            )
        }
    }

    override suspend fun consume(frame: FramePacket) {
        // Metadata-only path used by Day 1-2 tests to validate queue/drop behavior.
    }

    override fun resetInterpreter() {
        scope.launch(consumerDispatcher) {
            resetTrackingState()
            _poseResult.value = null
            lastFrameSkippedPose.set(true)
            clearPoseValidationQueue(recycleBitmaps = true)
            poseValidationActive.set(false)
            poseValidationComplete.set(false)
            closeInterpreterResources()
            closePoseResources()
        }
    }

    override fun shutdown() {
        frameChannel.close()
        resetTrackingState()
        _poseResult.value = null
        lastFrameSkippedPose.set(true)
        clearPoseValidationQueue(recycleBitmaps = true)
        poseValidationActive.set(false)
        poseValidationComplete.set(false)
        closeInterpreterResources()
        closePoseResources()
        poseValidationResultChannel.close()
        consumerJob.cancel()
        aggregateJob.cancel()
    }

    private fun initializePoseInterpreterIfNeeded() {
        if (poseInterpreter != null) return
        val poseBuffer = poseModelBufferProvider?.invoke() ?: return
        val useGpu = currentMode.get() != InferenceMode.CPU
        try {
            poseInterpreter = poseInterpreterFactory(poseBuffer, useGpu)
        } catch (error: Throwable) {
            _lastError.value =
                "Pose interpreter init failed: ${error.message ?: "unknown error"}"
            closePoseResources()
        }
    }

    override fun submitPoseValidationBatch(bitmaps: List<Bitmap>) {
        if (bitmaps.isEmpty()) return
        if (poseValidationActive.get()) {
            bitmaps.forEach { it.recycle() }
            _lastError.value = "Pose validation already running"
            return
        }

        clearPoseValidationQueue(recycleBitmaps = true)
        poseValidationComplete.set(false)
        bitmaps.forEachIndexed { index, bitmap ->
            poseValidationQueue.add(
                PoseValidationEntry(
                    sourceIndex = index,
                    sourceName = "frame_${(index + 1).toString().padStart(4, '0')}",
                    bitmap = bitmap
                )
            )
        }
        poseValidationActive.set(true)
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
        if (!switched && (target == InferenceMode.GPU || target == InferenceMode.NNAPI)) {
            val delegateFailure = _lastError.value
            switchInterpreter(InferenceMode.CPU)
            if (!delegateFailure.isNullOrBlank()) {
                _lastError.value = delegateFailure
            }
        }
        _isSwitchingMode.value = false
    }

    private fun switchInterpreter(mode: InferenceMode): Boolean {
        closeInterpreterResources()
        closePoseResources()
        resetTrackingState()

        val mappedModel = modelBufferProvider?.invoke()
        if (mappedModel == null) {
            _lastError.value = "Model buffer provider unavailable"
            currentMode.set(InferenceMode.CPU)
            return false
        }

        val options = Interpreter.Options().apply {
            setNumThreads(4)
            setAllowBufferHandleOutput(true)
        }

        var localGpuDelegate: GpuDelegate? = null
        var localNnApiDelegate: NnApiDelegate? = null

        when (mode) {
            InferenceMode.GPU -> {
                val compatibility = CompatibilityList()
                if (!compatibility.isDelegateSupportedOnThisDevice) {
                    _lastError.value = "GPU delegate unsupported on this device"
                    return false
                }
                localGpuDelegate = buildSustainedSpeedGpuDelegate().also {
                    options.addDelegate(it)
                }
            }
            InferenceMode.NNAPI -> {
                localNnApiDelegate = try {
                    NnApiDelegate().also { options.addDelegate(it) }
                } catch (error: Throwable) {
                    _lastError.value =
                        "NNAPI delegate init failed: ${error.message ?: "unknown error"}"
                    return false
                }
            }
            InferenceMode.CPU -> {
                // No delegate for CPU mode.
            }
        }

        return try {
            val localInterpreter = Interpreter(mappedModel, options)
            validateTensorContract(localInterpreter)
            interpreter = localInterpreter
            gpuDelegate = localGpuDelegate
            nnApiDelegate = localNnApiDelegate
            currentMode.set(mode)
            _lastError.value = null
            true
        } catch (error: Throwable) {
            localGpuDelegate?.close()
            localNnApiDelegate?.close()
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
        nnApiDelegate?.close()
        nnApiDelegate = null
    }

    private fun closePoseResources() {
        poseInterpreter?.close()
        poseInterpreter = null
    }

    private suspend fun processImage(image: ImageProxy) {
        try {
            if (interpreter == null) {
                maybeApplyPendingMode(force = true)
            }
            initializePoseInterpreterIfNeeded()
            val localInterpreter = interpreter ?: return
            val frameStartNs = System.nanoTime()
            perFrameTimingAccumulator.reset()

            val imageTimestampNs = image.imageInfo.timestamp
            val rotationDegrees = image.imageInfo.rotationDegrees
            val normalizedRotation = ((rotationDegrees % 360) + 360) % 360
            val deltaMs = if (lastImageTimestampNs > 0L) {
                (imageTimestampNs - lastImageTimestampNs) / 1_000_000.0
            } else {
                -1.0
            }
            lastImageTimestampNs = imageTimestampNs
            val mode = currentMode.get()
            val expectedTarget = expectedTargetRotation.get()
            val expectedRaw = expectedFrameRotationDegrees.get()

            updateRotationTelemetry(
                frameRotationDegrees = rotationDegrees,
                expectedTarget = expectedTarget,
                expectedRaw = expectedRaw
            )

            if (ROTATION_DEBUG_LOGS && normalizedRotation != lastNormalizedRotation) {
                if (lastNormalizedRotation != -1) {
                    Log.i(
                        TAG,
                        "[FRAME][ROT_CHANGE] tsNs=$imageTimestampNs deltaMs=${"%.3f".format(deltaMs)} " +
                            "rawRot=$rotationDegrees normRot=$normalizedRotation prevNormRot=$lastNormalizedRotation " +
                            "mode=$mode dropped=${droppedByOverflow.get()} queueDepth=${queueDepth.get()} " +
                            "expectedTarget=$expectedTarget expectedRaw=$expectedRaw"
                    )
                } else {
                    Log.i(
                        TAG,
                        "[FRAME][FIRST_ROT] tsNs=$imageTimestampNs deltaMs=NA " +
                            "rawRot=$rotationDegrees normRot=$normalizedRotation mode=$mode " +
                            "dropped=${droppedByOverflow.get()} queueDepth=${queueDepth.get()} " +
                            "expectedTarget=$expectedTarget expectedRaw=$expectedRaw"
                    )
                }
                lastNormalizedRotation = normalizedRotation
            }

            val sensorBitmap = image.toBitmap()
            val rotatedBitmap = rotateBitmapForDisplay(sensorBitmap, normalizedRotation)
            try {
                val rotatedWidth = rotatedBitmap.width
                val rotatedHeight = rotatedBitmap.height
                val yoloPreprocessStartNs = System.nanoTime()
                yoloTensorImage.load(rotatedBitmap)
                val tensorImage = yoloImageProcessor.process(yoloTensorImage)
                val yoloPreprocessMs = elapsedMs(yoloPreprocessStartNs)

                val inputBuffer = tensorImage.buffer
                val boxes: List<DetectionBox>
                val yoloInferenceMs: Double
                val yoloNmsMs: Double
                when (outputFormat) {
                    ModelOutputFormat.RAW_8400 -> {
                        for (row in outputTensorRaw[0]) row.fill(0f)
                        val yoloInferenceStartNs = System.nanoTime()
                        localInterpreter.run(inputBuffer, outputTensorRaw)
                        yoloInferenceMs = elapsedMs(yoloInferenceStartNs)

                        val yoloNmsStartNs = System.nanoTime()
                        boxes = parseModelOutput(outputTensorRaw, CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD)
                        yoloNmsMs = elapsedMs(yoloNmsStartNs)
                    }
                    ModelOutputFormat.END_TO_END_300 -> {
                        for (row in outputTensorE2E[0]) row.fill(0f)
                        val yoloInferenceStartNs = System.nanoTime()
                        localInterpreter.run(inputBuffer, outputTensorE2E)
                        yoloInferenceMs = elapsedMs(yoloInferenceStartNs)

                        val yoloNmsStartNs = System.nanoTime()
                        boxes = parseEndToEndOutput(outputTensorE2E, CONFIDENCE_THRESHOLD)
                        yoloNmsMs = elapsedMs(yoloNmsStartNs)
                    }
                }
                lastInferenceMs.set(yoloInferenceMs)
                var bestBallBox: DetectionBox? = null
                for (box in boxes) {
                    if (box.classId != BALL_CLASS_ID) continue
                    val currentBest = bestBallBox
                    if (currentBest == null || box.confidence > currentBest.confidence) {
                        bestBallBox = box
                    }
                }
                val dtSec = trackerDeltaSeconds(imageTimestampNs)
                val trackedBall = ballTracker.track(
                    dtSec = dtSec,
                    measurement = bestBallBox
                )
                latestTrackedBall.set(trackedBall.takeIf { it.isTracked })
                latestMissStreak.set(ballTracker.missFrames)
                _detections.value = DetectionFrame(
                    timestampNs = imageTimestampNs,
                    sourceWidth = rotatedWidth,
                    sourceHeight = rotatedHeight,
                    rotationDegrees = 0,
                    boxes = boxes,
                    trackedBall = trackedBall,
                    missStreak = ballTracker.missFrames,
                    // nanoTime shares CLOCK_MONOTONIC with Compose withFrameNanos for overlay-lag HUD
                    emitElapsedRealtimeNanos = System.nanoTime()
                )
                runLivePoseIfGated(bitmap = rotatedBitmap, boxes = boxes)
                processPoseValidationIfNeeded(
                    frameStartNs = frameStartNs,
                    yoloPreprocessMs = yoloPreprocessMs,
                    yoloInferenceMs = yoloInferenceMs,
                    yoloNmsMs = yoloNmsMs
                )
                appendPerFramePerfRow(
                    frameTotalMs = elapsedMs(frameStartNs),
                    yoloPreprocessMs = yoloPreprocessMs,
                    yoloInferenceMs = yoloInferenceMs,
                    yoloNmsMs = yoloNmsMs,
                    rotationDegrees = normalizedRotation
                )
            } finally {
                if (rotatedBitmap !== sensorBitmap) {
                    rotatedBitmap.recycle()
                }
                sensorBitmap.recycle()
            }
        } catch (error: Throwable) {
            _lastError.value = "Frame processing failed: ${error.message ?: "unknown error"}"
        } finally {
            image.close()
        }
    }

    private fun runLivePoseIfGated(bitmap: Bitmap, boxes: List<DetectionBox>) {
        perFrameTimingAccumulator.reset()
        val personBox = selectPersonBox(boxes) ?: run {
            _poseResult.value = null
            lastFrameSkippedPose.set(true)
            return
        }

        if (!shouldRunPose(boxes)) {
            _poseResult.value = null
            lastFrameSkippedPose.set(true)
            return
        }

        val localPoseInterpreter = poseInterpreter ?: run {
            _poseResult.value = null
            lastFrameSkippedPose.set(true)
            return
        }

        val poseCropStartNs = System.nanoTime()
        val personCrop = squarePadCrop(bitmap, personBox = personBox, marginFactor = 1.25f)
        val poseCropMs = elapsedMs(poseCropStartNs)
        try {
            val poseResult = localPoseInterpreter.infer(personCrop.bitmap)
            _poseResult.value = LivePoseOverlay(
                poseResult = poseResult,
                cropRectNormalized = personCrop.cropRectNormalized,
                // nanoTime shares CLOCK_MONOTONIC with Compose withFrameNanos for overlay-lag HUD
                emitElapsedRealtimeNanos = System.nanoTime()
            )
            perFrameTimingAccumulator.record(
                poseCropMs = poseCropMs,
                posePreprocessMs = poseResult.latency.preprocessMs,
                poseInferenceMs = poseResult.latency.inferenceMs,
                posePostprocessMs = poseResult.latency.postprocessMs
            )
            lastFrameSkippedPose.set(false)
        } catch (error: Throwable) {
            _poseResult.value = null
            perFrameTimingAccumulator.reset()
            lastFrameSkippedPose.set(true)
            _lastError.value = "Pose inference failed: ${error.message ?: "unknown error"}"
        } finally {
            personCrop.bitmap.recycle()
        }
    }

    private fun selectPersonBox(boxes: List<DetectionBox>): DetectionBox? {
        val persons = boxes.asSequence().filter { it.classId == PERSON_CLASS_ID }
        return when (personSelectionMode.get()) {
            PersonSelectionMode.HIGHEST_CONFIDENCE,
            PersonSelectionMode.REID_TRACKED -> {
                // ADR-006 stub: REID_TRACKED falls back to highest confidence in Phase 0.
                persons.maxByOrNull { it.confidence }
            }
        }
    }

    private fun shouldRunPose(boxes: List<DetectionBox>): Boolean {
        return when (poseGatingMode.get()) {
            PoseGatingMode.EVERY_FRAME_WITH_PERSON -> true
            PoseGatingMode.SHOOT_CLASS_GATED -> {
                // ADR-006 Day 6 stub: keep EVERY_FRAME behavior until shoot-gated wiring lands.
                @Suppress("UNUSED_VARIABLE")
                val hasShootSignal = boxes.any { it.classId == SHOOT_CLASS_ID }
                true
            }
            PoseGatingMode.FSM_GATED -> {
                // ADR-006 Day 6 stub: keep EVERY_FRAME behavior until Day 7 FSM wiring lands.
                true
            }
        }
    }

    private fun appendPerFramePerfRow(
        frameTotalMs: Double,
        yoloPreprocessMs: Double,
        yoloInferenceMs: Double,
        yoloNmsMs: Double,
        rotationDegrees: Int
    ) {
        val logger = perFrameLogger ?: return
        val poseSkipped = lastFrameSkippedPose.get()
        val row = PerFramePerfRow(
            timestampMs = System.currentTimeMillis(),
            frameTotalMs = frameTotalMs,
            yoloPreprocessMs = yoloPreprocessMs,
            yoloInferenceMs = yoloInferenceMs,
            yoloNmsMs = yoloNmsMs,
            poseCropMs = perFrameTimingAccumulator.poseCropMs,
            posePreprocessMs = perFrameTimingAccumulator.posePreprocessMs,
            poseInferenceMs = perFrameTimingAccumulator.poseInferenceMs,
            posePostprocessMs = perFrameTimingAccumulator.posePostprocessMs,
            poseSkipped = poseSkipped,
            ramMb = currentProcessRamMb(),
            thermalStatus = thermalStatusProvider(),
            fps1sWindow = _stats.value.analysisFps,
            rotationDegrees = rotationDegrees,
            mode = perFrameMode,
            device = perFrameDevice,
            gpuMode = currentMode.get().name
        )
        runCatching { logger.append(row) }
            .onFailure { error ->
                _lastError.value = "Per-frame logger append failed: ${error.message ?: "unknown error"}"
            }
    }

    @VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
    internal fun runLivePoseForTest(bitmap: Bitmap, boxes: List<DetectionBox>) {
        runLivePoseIfGated(bitmap, boxes)
    }

    @VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
    internal fun emitPerFrameRowForTest(
        frameTotalMs: Double,
        yoloPreprocessMs: Double,
        yoloInferenceMs: Double,
        yoloNmsMs: Double,
        rotationDegrees: Int = 0
    ) {
        appendPerFramePerfRow(
            frameTotalMs = frameTotalMs,
            yoloPreprocessMs = yoloPreprocessMs,
            yoloInferenceMs = yoloInferenceMs,
            yoloNmsMs = yoloNmsMs,
            rotationDegrees = rotationDegrees
        )
    }

    @VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
    internal fun initializePoseInterpreterForTest() {
        initializePoseInterpreterIfNeeded()
    }

    @VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
    internal fun forceCurrentInferenceModeForTest(mode: InferenceMode) {
        currentMode.set(mode)
    }

    @VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
    internal fun switchInterpreterForTest(mode: InferenceMode): Boolean = switchInterpreter(mode)

    @VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
    internal fun hasPoseInterpreterForTest(): Boolean = poseInterpreter != null

    private fun processPoseValidationIfNeeded(
        frameStartNs: Long,
        yoloPreprocessMs: Double,
        yoloInferenceMs: Double,
        yoloNmsMs: Double
    ) {
        if (!poseValidationActive.get()) return

        val entry = poseValidationQueue.poll() ?: run {
            poseValidationActive.set(false)
            poseValidationComplete.set(true)
            return
        }
        val localPoseInterpreter = poseInterpreter ?: run {
            clearPoseValidationQueue(recycleBitmaps = true)
            entry.bitmap.recycle()
            poseValidationActive.set(false)
            poseValidationComplete.set(true)
            _lastError.value = "Pose interpreter unavailable for validation"
            poseValidationResultChannel.trySend(
                PoseFrameResult(
                    sourceIndex = entry.sourceIndex,
                    sourceName = entry.sourceName,
                    yoloPreprocessMs = yoloPreprocessMs,
                    yoloInferenceMs = yoloInferenceMs,
                    yoloNmsMs = yoloNmsMs,
                    poseCropMs = 0.0,
                    posePreprocessMs = 0.0,
                    poseInferenceMs = 0.0,
                    posePostprocessMs = 0.0,
                    frameTotalMs = elapsedMs(frameStartNs),
                    posePresence = 0f,
                    visibleJoints33 = 0,
                    decodedLandmarks39 = 0,
                    imageLandmarks33 = emptyList(),
                    status = "error",
                    error = "Pose interpreter unavailable",
                    isLast = true
                )
            )
            return
        }

        var poseCropMs = 0.0
        var posePreprocessMs = 0.0
        var poseInferenceMs = 0.0
        var posePostprocessMs = 0.0
        var posePresence = 0f
        var visibleJoints33 = 0
        var decodedLandmarks39 = 0
        var imageLandmarks33: List<PoseImageLandmark> = emptyList()
        var status = "ok"
        var error: String? = null

        try {
            val poseCropStartNs = System.nanoTime()
            val validationBitmap = squarePadCrop(entry.bitmap)
            poseCropMs = elapsedMs(poseCropStartNs)

            val poseResult = try {
                localPoseInterpreter.infer(validationBitmap)
            } finally {
                if (validationBitmap !== entry.bitmap) {
                    validationBitmap.recycle()
                }
            }
            posePreprocessMs = poseResult.latency.preprocessMs
            poseInferenceMs = poseResult.latency.inferenceMs
            posePostprocessMs = poseResult.latency.postprocessMs
            posePresence = poseResult.posePresence
            imageLandmarks33 = poseResult.imageLandmarks33
            visibleJoints33 = poseResult.imageLandmarks33.count {
                it.visibility > POSE_VISIBILITY_THRESHOLD && it.presence > POSE_PRESENCE_THRESHOLD
            }
            decodedLandmarks39 = poseResult.imageLandmarks39.size
        } catch (throwable: Throwable) {
            status = "error"
            error = throwable.message ?: "unknown error"
        } finally {
            entry.bitmap.recycle()
        }

        val isLast = poseValidationQueue.isEmpty()
        if (isLast) {
            poseValidationActive.set(false)
            poseValidationComplete.set(true)
        }

        poseValidationResultChannel.trySend(
            PoseFrameResult(
                sourceIndex = entry.sourceIndex,
                sourceName = entry.sourceName,
                yoloPreprocessMs = yoloPreprocessMs,
                yoloInferenceMs = yoloInferenceMs,
                yoloNmsMs = yoloNmsMs,
                poseCropMs = poseCropMs,
                posePreprocessMs = posePreprocessMs,
                poseInferenceMs = poseInferenceMs,
                posePostprocessMs = posePostprocessMs,
                frameTotalMs = elapsedMs(frameStartNs),
                posePresence = posePresence,
                visibleJoints33 = visibleJoints33,
                decodedLandmarks39 = decodedLandmarks39,
                imageLandmarks33 = imageLandmarks33,
                status = status,
                error = error,
                isLast = isLast
            )
        )
    }

    private fun clearPoseValidationQueue(recycleBitmaps: Boolean) {
        while (true) {
            val next = poseValidationQueue.poll() ?: break
            if (recycleBitmaps) {
                next.bitmap.recycle()
            }
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

    private fun updateRotationTelemetry(
        frameRotationDegrees: Int,
        expectedTarget: Int,
        expectedRaw: Int
    ) {
        val dropped = droppedByOverflow.get()
        val nowMs = SystemClock.elapsedRealtime()

        if (expectedRaw !in VALID_ROTATIONS_DEGREES) {
            rotationMismatchStartElapsedMs = -1L
            _rotationTelemetry.value = RotationTelemetry(
                expectedTargetRotation = expectedTarget,
                expectedFrameRotationDegrees = expectedRaw,
                frameRotationDegrees = frameRotationDegrees,
                rotationMismatchMs = 0L,
                droppedFramesSnapshot = dropped,
                stallState = RotationStallState.NONE
            )
            return
        }

        if (frameRotationDegrees == expectedRaw) {
            val previousMismatchMs = if (rotationMismatchStartElapsedMs >= 0L) {
                nowMs - rotationMismatchStartElapsedMs
            } else {
                0L
            }
            if (rotationMismatchStartElapsedMs >= 0L && previousMismatchMs >= RECONCILE_THRESHOLD_MS) {
                Log.i(
                    TAG,
                    "[FRAME][ROT_STALL_RESOLVED] mismatchMs=$previousMismatchMs frameRaw=$frameRotationDegrees expectedRaw=$expectedRaw expectedTarget=$expectedTarget dropped=$dropped"
                )
            }
            rotationMismatchStartElapsedMs = -1L
            _rotationTelemetry.value = RotationTelemetry(
                expectedTargetRotation = expectedTarget,
                expectedFrameRotationDegrees = expectedRaw,
                frameRotationDegrees = frameRotationDegrees,
                rotationMismatchMs = 0L,
                droppedFramesSnapshot = dropped,
                stallState = RotationStallState.NONE
            )
            return
        }

        if (rotationMismatchStartElapsedMs < 0L) {
            rotationMismatchStartElapsedMs = nowMs
        }

        val mismatchMs = nowMs - rotationMismatchStartElapsedMs
        val stallState = when {
            dropped != 0L -> RotationStallState.MISMATCH
            mismatchMs >= RECOVERY_THRESHOLD_MS -> RotationStallState.RECOVERY_REQUESTED
            mismatchMs >= RECONCILE_THRESHOLD_MS -> RotationStallState.RECONCILE
            else -> RotationStallState.MISMATCH
        }

        _rotationTelemetry.value = RotationTelemetry(
            expectedTargetRotation = expectedTarget,
            expectedFrameRotationDegrees = expectedRaw,
            frameRotationDegrees = frameRotationDegrees,
            rotationMismatchMs = mismatchMs,
            droppedFramesSnapshot = dropped,
            stallState = stallState
        )

        if (dropped == 0L &&
            mismatchMs >= RECONCILE_THRESHOLD_MS &&
            (lastRotationStallLogElapsedMs < 0L || nowMs - lastRotationStallLogElapsedMs >= STALL_LOG_INTERVAL_MS)
        ) {
            Log.w(
                TAG,
                "[FRAME][ROT_STALL] mismatchMs=$mismatchMs frameRaw=$frameRotationDegrees expectedRaw=$expectedRaw expectedTarget=$expectedTarget dropped=$dropped stallState=$stallState"
            )
            lastRotationStallLogElapsedMs = nowMs
        }
    }

    private fun trackerDeltaSeconds(frameTimestampNs: Long): Float {
        val previousTimestampNs = lastTrackerTimestampNs
        lastTrackerTimestampNs = frameTimestampNs
        if (previousTimestampNs <= 0L || frameTimestampNs <= previousTimestampNs) {
            return DEFAULT_TRACKER_DT_SEC
        }
        val deltaNs = frameTimestampNs - previousTimestampNs
        return (deltaNs / 1_000_000_000f).coerceIn(MIN_TRACKER_DT_SEC, MAX_TRACKER_DT_SEC)
    }

    private fun resetTrackingState() {
        ballTracker.reset()
        latestTrackedBall.set(null)
        latestMissStreak.set(0)
        lastFrameSkippedPose.set(true)
        lastImageTimestampNs = -1L
        lastTrackerTimestampNs = -1L
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
        val trackedBall = latestTrackedBall.get()

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
                thermalStatus = thermalStatusProvider(),
                trackingActive = trackedBall?.isTracked == true,
                trackCx = trackedBall?.centroidX?.toDouble(),
                trackCy = trackedBall?.centroidY?.toDouble(),
                trackVx = trackedBall?.velocityX?.toDouble(),
                trackVy = trackedBall?.velocityY?.toDouble(),
                missStreak = latestMissStreak.get(),
                poseSkipped = lastFrameSkippedPose.get()
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

    private fun elapsedMs(startNs: Long): Double = (System.nanoTime() - startNs) / 1_000_000.0

    private class PerFrameTimingAccumulator {
        var poseCropMs: Double = 0.0
            private set
        var posePreprocessMs: Double = 0.0
            private set
        var poseInferenceMs: Double = 0.0
            private set
        var posePostprocessMs: Double = 0.0
            private set

        fun reset() {
            poseCropMs = 0.0
            posePreprocessMs = 0.0
            poseInferenceMs = 0.0
            posePostprocessMs = 0.0
        }

        fun record(
            poseCropMs: Double,
            posePreprocessMs: Double,
            poseInferenceMs: Double,
            posePostprocessMs: Double
        ) {
            this.poseCropMs = poseCropMs
            this.posePreprocessMs = posePreprocessMs
            this.poseInferenceMs = poseInferenceMs
            this.posePostprocessMs = posePostprocessMs
        }
    }

    private data class PoseValidationEntry(
        val sourceIndex: Int,
        val sourceName: String,
        val bitmap: Bitmap
    )

    private sealed interface FrameTask {
        data class Metadata(val frame: FramePacket) : FrameTask
        data class Image(val image: ImageProxy) : FrameTask
    }

    companion object {
        private const val TAG = "CVRotation"
        private const val ROTATION_DEBUG_LOGS = false
        private const val RECONCILE_THRESHOLD_MS = 2_000L
        private const val RECOVERY_THRESHOLD_MS = 5_000L
        private const val STALL_LOG_INTERVAL_MS = 1_000L
        private const val MODEL_INPUT_SIZE = 640
        private const val OUTPUT_BOXES = 8400
        private const val CONFIDENCE_THRESHOLD = 0.40f
        private const val NMS_IOU_THRESHOLD = 0.50f
        private const val POSE_VISIBILITY_THRESHOLD = 0.60f
        private const val POSE_PRESENCE_THRESHOLD = 0.50f
        private const val BALL_CLASS_ID = 0
        private const val PERSON_CLASS_ID = 2
        private const val SHOOT_CLASS_ID = 4
        private const val DEFAULT_TRACKER_DT_SEC = 1f / 30f
        private const val MIN_TRACKER_DT_SEC = 1f / 120f
        private const val MAX_TRACKER_DT_SEC = 0.25f

        private const val E2E_MAX_DETS = 300
        private const val E2E_FIELDS = 6

        private val VALID_ROTATIONS_DEGREES = setOf(0, 90, 180, 270)
        private val CUSTOM_CLASS_NAMES = arrayOf("ball", "made", "person", "rim", "shoot")
    }
}
