package com.courtvision.spike.camera

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Build
import android.os.PowerManager
import android.provider.OpenableColumns
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.courtvision.spike.pipeline.FrameProcessor
import com.courtvision.spike.pipeline.FrameProcessorGateway
import com.courtvision.spike.pipeline.GpuDelegateProbe
import com.courtvision.spike.pipeline.GpuProbeResult
import com.courtvision.spike.pipeline.DetectionFrame
import com.courtvision.spike.pipeline.InferenceMode
import com.courtvision.spike.pipeline.NnApiDelegateProbe
import com.courtvision.spike.pipeline.PerformanceCsvLogger
import com.courtvision.spike.pipeline.PerformanceLogger
import com.courtvision.spike.pipeline.PerFramePerfCsvLogger
import com.courtvision.spike.pipeline.PerFramePerfLogger
import com.courtvision.spike.pipeline.PerFramePerfRow
import com.courtvision.spike.pipeline.PoseFrameResult
import com.courtvision.spike.pipeline.PoseImageLandmark
import com.courtvision.spike.pipeline.PoseTensorContract
import com.courtvision.spike.pipeline.QnnDelegateProbe
import com.courtvision.spike.pipeline.QnnProbeResult
import com.courtvision.spike.pipeline.QnnStatus
import com.courtvision.spike.pipeline.RotationTelemetry
import com.courtvision.spike.pipeline.SpikeImageAnalyzer
import com.courtvision.spike.pipeline.TrackingCsvLogger
import com.courtvision.spike.pipeline.TrackingLogger
import java.io.File
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.floor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.transformWhile
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class CameraViewModel(
    application: Application
) : AndroidViewModel(application) {

    private val overrides = testOverrides
    private val availableModelPaths: List<String> = overrides?.modelPaths ?: discoverModelAssets()
    @Volatile
    private var selectedModelPath: String =
        overrides?.initialModelPath
            ?.takeIf { availableModelPaths.contains(it) }
            ?: chooseInitialModel(availableModelPaths)

    private val performanceLogger: PerformanceLogger =
        overrides?.performanceLogger ?: PerformanceCsvLogger(application)
    private val trackingLogger: TrackingLogger =
        overrides?.trackingLogger ?: TrackingCsvLogger(application)
    private val perFramePerfLogger: PerFramePerfLogger =
        overrides?.perFramePerfLogger ?: if (overrides != null) {
            NoOpPerFramePerfLogger
        } else {
            PerFramePerfCsvLogger(application)
        }
    private val powerManager: PowerManager? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            application.getSystemService(PowerManager::class.java)
        } else {
            null
        }
    private val frameProcessor: FrameProcessorGateway =
        overrides?.frameProcessor ?: FrameProcessor(
            scope = viewModelScope,
            modelBufferProvider = ::loadModelBuffer,
            poseModelBufferProvider = ::loadPoseModelBuffer,
            nativeLibraryDir = application.applicationInfo.nativeLibraryDir,
            modelCacheDir = application.cacheDir.absolutePath,
            thermalStatusProvider = {
                if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
                    "UNKNOWN"
                } else {
                    mapThermalStatus(powerManager?.currentThermalStatus)
                }
            },
            perFrameLogger = perFramePerfLogger,
            perFrameMode = PER_FRAME_MODE,
            perFrameDevice = Build.MODEL
        )

    private val _uiState = MutableStateFlow(
        CameraUiState(
            availableModels = availableModelPaths,
            selectedModel = selectedModelPath,
            logFilePath = performanceLogger.filePath,
            trackingLogFilePath = trackingLogger.filePath
        )
    )
    private val analyzer = SpikeImageAnalyzer(frameProcessor) {
        _uiState.value.modelConfirmed
    }
    val uiState: StateFlow<CameraUiState> = _uiState.asStateFlow()
    val rotationTelemetry: StateFlow<RotationTelemetry> = frameProcessor.rotationTelemetry

    @Volatile
    private var cachedSelectedModelBuffer: MappedByteBuffer? = null
    @Volatile
    private var cachedQnnModelBuffer: MappedByteBuffer? = null
    @Volatile
    private var cachedPoseModelBuffer: MappedByteBuffer? = null

    private var poseValidationCollectorJob: Job? = null

    init {
        // Phase 0 decision: keep probes synchronous in init for deterministic startup behavior.
        // Phase 2 note: move delegate probes to Dispatchers.Default to avoid any JNI work on Main.
        val probeResult = overrides?.gpuProbeResult ?: GpuDelegateProbe.probe()
        val qnnResult = overrides?.qnnProbeResult ?: QnnDelegateProbe.probe()
        val nnApiResult = overrides?.nnApiProbeResult ?: NnApiDelegateProbe.probe()
        _uiState.update {
            it.copy(
                gpuProbeResult = probeResult,
                qnnProbeResult = qnnResult,
                qnnAvailable = qnnResult.status == QnnStatus.QNN_SUPPORTED &&
                    qnnResult.htpQuantizedSupported,
                nnApiProbeResult = nnApiResult,
                nnApiAvailable = nnApiResult.startsWith("NNAPI_AVAILABLE")
            )
        }
        frameProcessor.setTrackerMaxMissFrames(_uiState.value.trackerMaxMissFrames)
        frameProcessor.setTrackerNoise(
            _uiState.value.trackerProcessNoise,
            _uiState.value.trackerMeasurementNoise
        )

        viewModelScope.launch {
            frameProcessor.stats.collect { stats ->
                _uiState.update { state ->
                    state.copy(
                        stats = stats,
                        selectedMode = stats.delegateMode
                    )
                }
                performanceLogger.append(
                    stats = stats,
                    gpuStatus = _uiState.value.gpuProbeResult.status,
                    modelUsed = selectedModelPath
                )
            }
        }

        viewModelScope.launch {
            frameProcessor.detections.collect { frame ->
                val isModelConfirmed = _uiState.value.modelConfirmed
                _uiState.update { state ->
                    if (isModelConfirmed) {
                        state.copy(detectionFrame = frame)
                    } else {
                        state.copy(detectionFrame = DetectionFrame())
                    }
                }
                if (isModelConfirmed) {
                    trackingLogger.append(
                        frame = frame,
                        delegateMode = _uiState.value.selectedMode,
                        modelUsed = selectedModelPath
                    )
                }
            }
        }

        viewModelScope.launch {
            frameProcessor.poseResult.collect { overlay ->
                _uiState.update { state ->
                    if (state.modelConfirmed) {
                        state.copy(poseOverlay = overlay)
                    } else {
                        state.copy(poseOverlay = null)
                    }
                }
            }
        }

        viewModelScope.launch {
            frameProcessor.isSwitchingMode.collect { switching ->
                _uiState.update { state ->
                    state.copy(isSwitchingMode = switching)
                }
            }
        }

        viewModelScope.launch {
            frameProcessor.lastError.collect { pipelineError ->
                if (pipelineError != null) {
                    _uiState.update { state ->
                        state.copy(lastError = pipelineError)
                    }
                }
            }
        }
    }

    fun imageAnalyzer(): ImageAnalysis.Analyzer = analyzer

    fun onCameraPermissionResult(granted: Boolean) {
        _uiState.update { state ->
            val nextStatus = when {
                !granted -> CameraStatus.PERMISSION_REQUIRED
                state.cameraStatus == CameraStatus.ERROR -> CameraStatus.ERROR
                else -> state.cameraStatus
            }
            state.copy(
                hasCameraPermission = granted,
                cameraStatus = nextStatus
            )
        }
    }

    fun onCameraStarted() {
        _uiState.update {
            it.copy(
                cameraStatus = CameraStatus.RUNNING,
                lastError = null
            )
        }
    }

    fun onCameraStopped() {
        _uiState.update {
            if (it.hasCameraPermission) {
                it.copy(cameraStatus = CameraStatus.IDLE)
            } else {
                it.copy(cameraStatus = CameraStatus.PERMISSION_REQUIRED)
            }
        }
    }

    fun onCameraError(message: String) {
        _uiState.update {
            it.copy(
                cameraStatus = CameraStatus.ERROR,
                lastError = message
            )
        }
    }

    fun setInferenceMode(mode: InferenceMode) {
        frameProcessor.setInferenceMode(mode)
        _uiState.update {
            it.copy(
                selectedMode = mode,
                isSwitchingMode = true
            )
        }
    }

    fun setTrackerMaxMissFrames(value: Int) {
        val normalized = value.coerceIn(1, 30)
        frameProcessor.setTrackerMaxMissFrames(normalized)
        _uiState.update { it.copy(trackerMaxMissFrames = normalized) }
    }

    fun setTrackerNoise(processNoise: Float, measurementNoise: Float) {
        frameProcessor.setTrackerNoise(processNoise, measurementNoise)
        _uiState.update {
            it.copy(
                trackerProcessNoise = processNoise,
                trackerMeasurementNoise = measurementNoise
            )
        }
    }

    fun updateExpectedRotation(
        expectedTargetRotation: Int,
        expectedFrameRotationDegrees: Int,
        source: String
    ) {
        frameProcessor.updateExpectedRotation(
            expectedTargetRotation = expectedTargetRotation,
            expectedFrameRotationDegrees = expectedFrameRotationDegrees,
            source = source
        )
    }

    fun rotationTelemetrySnapshot(): RotationTelemetry = frameProcessor.rotationTelemetry.value

    fun setModel(modelPath: String) {
        if (isModelSelectionLocked()) return
        if (!availableModelPaths.contains(modelPath)) return
        if (modelPath == selectedModelPath) return

        if (_uiState.value.poseValidationRunning) {
            cancelPoseValidation(
                status = "CANCELLED: model changed",
                error = "Pose validation cancelled because model changed before start."
            )
        }
        selectedModelPath = modelPath
        cachedSelectedModelBuffer = null
        frameProcessor.resetInterpreter()

        _uiState.update {
            it.copy(
                selectedModel = modelPath,
                detectionFrame = DetectionFrame(),
                poseOverlay = null,
                lastError = null
            )
        }
    }

    fun confirmModel() {
        _uiState.update { state ->
            if (state.modelConfirmed) {
                state
            } else {
                state.copy(modelConfirmed = true)
            }
        }
    }

    fun restartSession() {
        poseValidationCollectorJob?.cancel()
        poseValidationCollectorJob = null
        cachedSelectedModelBuffer = null
        cachedQnnModelBuffer = null
        frameProcessor.resetInterpreter()
        _uiState.update { state ->
            state.copy(
                modelConfirmed = false,
                detectionFrame = DetectionFrame(),
                poseOverlay = null,
                isSwitchingMode = false,
                poseValidationRunning = false,
                poseValidationStatus = "IDLE",
                poseValidationOutputPath = "",
                lastError = null
            )
        }
    }

    fun startDay5PoseIsolatedValidation(uris: List<Uri>) {
        if (_uiState.value.poseValidationRunning) return
        if (!_uiState.value.modelConfirmed) {
            _uiState.update {
                it.copy(
                    poseValidationStatus = "FAILED: start inference first",
                    lastError = "Confirm model and start inference before pose validation."
                )
            }
            return
        }
        if (uris.isEmpty()) {
            _uiState.update {
                it.copy(
                    poseValidationStatus = "FAILED: no images selected",
                    lastError = "No images selected for pose validation."
                )
            }
            return
        }

        poseValidationCollectorJob?.cancel()
        poseValidationCollectorJob = null

        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    poseValidationRunning = true,
                    poseValidationStatus = "RUNNING",
                    poseValidationOutputPath = "",
                    lastError = null
                )
            }

            val outputDir = createPoseOutputDirectory()
            if (outputDir == null) {
                _uiState.update {
                    it.copy(
                        poseValidationRunning = false,
                        poseValidationStatus = "FAILED: cannot create output directory",
                        lastError = "Unable to create pose output directory."
                    )
                }
                return@launch
            }

            val decodedSources = withContext(Dispatchers.IO) {
                decodePoseValidationSources(uris)
            }
            if (decodedSources.isEmpty()) {
                _uiState.update {
                    it.copy(
                        poseValidationRunning = false,
                        poseValidationStatus = "FAILED: no decodable images",
                        poseValidationOutputPath = outputDir.absolutePath,
                        lastError = "Unable to decode selected images for pose validation."
                    )
                }
                return@launch
            }

            val csvFile = File(outputDir, "day5_pose_validation.csv")
            val summaryFile = File(outputDir, "day5_pose_summary.txt")
            csvFile.writeText(POSE_VALIDATION_CSV_HEADER + "\n")

            val sourcesByIndex = decodedSources.mapIndexed { index, source ->
                index to source.source
            }.toMap()

            poseValidationCollectorJob = viewModelScope.launch(Dispatchers.IO) {
                collectPoseValidationResults(
                    outputDir = outputDir,
                    csvFile = csvFile,
                    summaryFile = summaryFile,
                    sourcesByIndex = sourcesByIndex
                )
            }

            frameProcessor.submitPoseValidationBatch(decodedSources.map { it.bitmap })
        }
    }

    override fun onCleared() {
        poseValidationCollectorJob?.cancel()
        performanceLogger.close()
        trackingLogger.close()
        perFramePerfLogger.close()
        frameProcessor.shutdown()
        super.onCleared()
    }

    private fun isModelSelectionLocked(): Boolean = _uiState.value.modelConfirmed

    private fun cancelPoseValidation(status: String, error: String?) {
        poseValidationCollectorJob?.cancel()
        poseValidationCollectorJob = null
        _uiState.update { state ->
            state.copy(
                poseValidationRunning = false,
                poseValidationStatus = status,
                lastError = error
            )
        }
    }

    private fun loadModelBuffer(mode: InferenceMode): MappedByteBuffer {
        val assetPath = modelAssetPathForMode(mode)
        val cachedBuffer = if (mode == InferenceMode.QNN_NPU) {
            cachedQnnModelBuffer
        } else {
            cachedSelectedModelBuffer
        }
        cachedBuffer?.let { return it }

        synchronized(this) {
            val existingBuffer = if (mode == InferenceMode.QNN_NPU) {
                cachedQnnModelBuffer
            } else {
                cachedSelectedModelBuffer
            }
            existingBuffer?.let { return it }

            val mapped = getApplication<Application>().assets.openFd(assetPath).use { assetFile ->
                FileInputStream(assetFile.fileDescriptor).channel.use { channel ->
                    channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        assetFile.startOffset,
                        assetFile.declaredLength
                    )
                }
            }
            if (mode == InferenceMode.QNN_NPU) {
                cachedQnnModelBuffer = mapped
            } else {
                cachedSelectedModelBuffer = mapped
            }
            return mapped
        }
    }

    internal fun modelAssetPathForMode(mode: InferenceMode): String {
        return when (mode) {
            InferenceMode.QNN_NPU -> QNN_MODEL_ASSET_PATH
            InferenceMode.CPU,
            InferenceMode.GPU,
            InferenceMode.NNAPI -> selectedModelPath
        }
    }

    private fun loadPoseModelBuffer(): MappedByteBuffer {
        cachedPoseModelBuffer?.let { return it }
        synchronized(this) {
            cachedPoseModelBuffer?.let { return it }
            val mapped = getApplication<Application>().assets.openFd(POSE_MODEL_ASSET_PATH).use { assetFile ->
                FileInputStream(assetFile.fileDescriptor).channel.use { channel ->
                    channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        assetFile.startOffset,
                        assetFile.declaredLength
                    )
                }
            }
            cachedPoseModelBuffer = mapped
            return mapped
        }
    }

    private suspend fun collectPoseValidationResults(
        outputDir: File,
        csvFile: File,
        summaryFile: File,
        sourcesByIndex: Map<Int, PoseValidationSource>
    ) {
        val poseInferenceSamples = mutableListOf<Double>()
        val overlaySamples = mutableListOf<Double>()
        var successCount = 0
        var failureCount = 0

        try {
            frameProcessor.poseValidationResults
                .transformWhile { result ->
                    emit(result)
                    !result.isLast
                }
                .collect { result ->
                    val source = sourcesByIndex[result.sourceIndex]
                    val sourceName = source?.displayName ?: result.sourceName
                    val overlayMs = writePoseOverlayForResult(outputDir, source, result)

                    csvFile.appendText(
                        buildPoseCsvRow(
                            sourceName = sourceName,
                            overlayMs = overlayMs,
                            result = result
                        ) + "\n"
                    )

                    if (result.status == "ok") {
                        successCount += 1
                        poseInferenceSamples += result.poseInferenceMs
                    } else {
                        failureCount += 1
                    }
                    overlaySamples += overlayMs
                }

            val p95Inference = percentile(poseInferenceSamples, 0.95)
            val selectedMode = _uiState.value.selectedMode
            val gpuModeForSummary = when (selectedMode) {
                InferenceMode.CPU -> "CPU"
                InferenceMode.GPU -> "GPU"
                InferenceMode.NNAPI -> "NNAPI"
                InferenceMode.QNN_NPU -> "QNN_NPU"
            }
            val thermalStatusForSummary = readThermalStatus()
            val summaryLines = listOf(
                "device=${Build.MODEL}",
                "gpu_mode=$gpuModeForSummary",
                "input_size=${PoseTensorContract.INPUT_SIZE}",
                "thermal_status=$thermalStatusForSummary",
                "samples=${successCount + failureCount}",
                "success=$successCount",
                "failure=$failureCount",
                "pose_inference_p50_ms=${formatDouble(percentile(poseInferenceSamples, 0.50))}",
                "pose_inference_p95_ms=${formatDouble(p95Inference)}",
                "overlay_p50_ms=${formatDouble(percentile(overlaySamples, 0.50))}",
                "overlay_p95_ms=${formatDouble(percentile(overlaySamples, 0.95))}",
                "pose_inference_gate_s22=${formatGate(p95Inference)}",
                "mode=sequential_yolo_warm"
            )
            summaryFile.writeText(summaryLines.joinToString(separator = "\n"))

            _uiState.update {
                it.copy(
                    poseValidationRunning = false,
                    poseValidationStatus = "DONE: $successCount ok, $failureCount failed, p95 inf=${formatDouble(p95Inference)}ms",
                    poseValidationOutputPath = outputDir.absolutePath,
                    lastError = null
                )
            }
        } catch (_: CancellationException) {
            // Expected when session is restarted or ViewModel is cleared.
        } catch (error: Throwable) {
            Log.e(POSE_RUN_TAG, "Pose validation collection failed", error)
            _uiState.update {
                it.copy(
                    poseValidationRunning = false,
                    poseValidationStatus = "FAILED: ${error.message ?: "unknown error"}",
                    poseValidationOutputPath = outputDir.absolutePath,
                    lastError = "Pose validation failed: ${error.message ?: "unknown error"}"
                )
            }
        }
    }

    private suspend fun decodePoseValidationSources(uris: List<Uri>): List<PoseValidationInput> {
        return withContext(Dispatchers.IO) {
            val sources = mutableListOf<PoseValidationInput>()
            uris.forEachIndexed { index, uri ->
                val bitmap = decodeBitmap(uri) ?: return@forEachIndexed
                sources += PoseValidationInput(
                    source = PoseValidationSource(
                        uri = uri,
                        displayName = resolveDisplayName(uri, index)
                    ),
                    bitmap = bitmap
                )
            }
            sources
        }
    }

    private fun decodeBitmap(uri: Uri): Bitmap? {
        return try {
            getApplication<Application>().contentResolver.openInputStream(uri)?.use { stream ->
                val options = BitmapFactory.Options().apply {
                    // Force software-backed pixels for TensorImage.load()/getPixels() compatibility.
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                }
                BitmapFactory.decodeStream(stream, null, options)
            }
        } catch (_: Throwable) {
            null
        }
    }

    private fun resolveDisplayName(uri: Uri, index: Int): String {
        getApplication<Application>().contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME),
            null,
            null,
            null
        )?.use { cursor ->
            val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (nameIndex >= 0 && cursor.moveToFirst()) {
                val value = cursor.getString(nameIndex)
                if (!value.isNullOrBlank()) return value
            }
        }
        return "frame_${(index + 1).toString().padStart(4, '0')}.png"
    }

    private fun createPoseOutputDirectory(): File? {
        val root = overrides?.poseOutputRootProvider?.invoke()
            ?: getApplication<Application>().getExternalFilesDir(null)
            ?: return null
        val parent = File(root, POSE_OUTPUT_FOLDER_NAME)
        if (!parent.exists() && !parent.mkdirs()) return null
        val outputDir = File(parent, "inference_${timestampForFileName()}")
        if (!outputDir.exists() && !outputDir.mkdirs()) return null
        return outputDir
    }

    private fun writePoseOverlayForResult(
        outputDir: File,
        source: PoseValidationSource?,
        result: PoseFrameResult
    ): Double {
        if (source == null || result.status != "ok") return 0.0
        val sourceBitmap = decodeBitmap(source.uri) ?: return 0.0
        return try {
            val fileStem = sanitizeFileStem(source.displayName)
            renderPoseOverlay(
                source = sourceBitmap,
                landmarks = result.imageLandmarks33,
                outputFile = File(outputDir, "${fileStem}_overlay.png")
            )
        } finally {
            sourceBitmap.recycle()
        }
    }

    private fun buildPoseCsvRow(
        sourceName: String,
        overlayMs: Double,
        result: PoseFrameResult
    ): String {
        return buildString {
            append(timestampForCsv()).append(',')
            append(sanitizeCsv(sourceName)).append(',')
            append(formatDouble(result.yoloPreprocessMs)).append(',')
            append(formatDouble(result.yoloInferenceMs)).append(',')
            append(formatDouble(result.yoloNmsMs)).append(',')
            append(formatDouble(result.poseCropMs)).append(',')
            append(formatDouble(result.posePreprocessMs)).append(',')
            append(formatDouble(result.poseInferenceMs)).append(',')
            append(formatDouble(result.posePostprocessMs)).append(',')
            append(formatDouble(result.frameTotalMs)).append(',')
            append(formatDouble(overlayMs)).append(',')
            append(formatDouble(result.posePresence.toDouble())).append(',')
            append(result.visibleJoints33).append(',')
            append(result.decodedLandmarks39).append(',')
            append(result.status).append(',')
            append(sanitizeCsv(result.error.orEmpty()))
        }
    }

    private fun renderPoseOverlay(
        source: Bitmap,
        landmarks: List<PoseImageLandmark>,
        outputFile: File
    ): Double {
        val startNs = System.nanoTime()

        val mutable = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutable)
        val pointPaint = Paint().apply {
            color = Color.YELLOW
            style = Paint.Style.FILL
            strokeWidth = 5f
            isAntiAlias = true
        }
        val lowConfidencePaint = Paint().apply {
            color = Color.RED
            style = Paint.Style.FILL
            strokeWidth = 5f
            isAntiAlias = true
        }

        val side = maxOf(mutable.width, mutable.height).toFloat()
        val inputSize = PoseTensorContract.INPUT_SIZE.toFloat()
        val sideScale = side / inputSize
        val padLeft = (side - mutable.width) / 2f
        val padTop = (side - mutable.height) / 2f

        landmarks.forEach { landmark ->
            val x = (landmark.xPx * sideScale) - padLeft
            val y = (landmark.yPx * sideScale) - padTop
            val paint = if (landmark.visibility > VISIBILITY_THRESHOLD) pointPaint else lowConfidencePaint
            canvas.drawCircle(x, y, 4f, paint)
        }

        outputFile.outputStream().use { stream ->
            mutable.compress(Bitmap.CompressFormat.PNG, 100, stream)
        }
        mutable.recycle()

        return elapsedMs(startNs)
    }

    private fun sanitizeFileStem(fileName: String): String {
        val stem = fileName.substringBeforeLast('.', fileName)
        val sanitized = stem.replace(NON_FILE_STEM_CHARS_REGEX, "_")
        return sanitized.ifBlank { "frame" }
    }

    private fun discoverModelAssets(): List<String> {
        val assetManager = getApplication<Application>().assets
        val candidates = mutableSetOf<String>()
        val rootEntries = assetManager.list("").orEmpty()

        rootEntries.forEach { entry ->
            if (isSupportedModelAsset(entry)) {
                candidates.add(entry)
            }

            if (entry.endsWith("_saved_model")) {
                val dirEntries = assetManager.list(entry).orEmpty()
                dirEntries.forEach { fileName ->
                    val candidatePath = "$entry/$fileName"
                    if (isSupportedModelAsset(candidatePath)) {
                        candidates.add(candidatePath)
                    }
                }
            }
        }

        val filtered = candidates
            .asSequence()
            .sorted()
            .toList()

        return if (filtered.isEmpty()) {
            listOf(DEFAULT_MODEL_ASSET_PATH)
        } else {
            filtered
        }
    }

    private fun isSupportedModelAsset(path: String): Boolean {
        return path.endsWith("_float16.tflite") || path.endsWith("_float32.tflite")
    }

    private fun chooseInitialModel(models: List<String>): String {
        return when {
            models.contains(DEFAULT_MODEL_ASSET_PATH) -> DEFAULT_MODEL_ASSET_PATH
            models.isNotEmpty() -> models.first()
            else -> DEFAULT_MODEL_ASSET_PATH
        }
    }

    private fun timestampForFileName(): String =
        SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())

    private fun timestampForCsv(): String =
        SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(Date())

    private fun sanitizeCsv(value: String): String =
        value.replace(',', ';').replace('\n', ' ').replace('\r', ' ')

    private fun formatDouble(value: Double): String = String.format(Locale.US, "%.3f", value)

    private fun percentile(values: List<Double>, percentile: Double): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val index = floor((sorted.size - 1) * percentile).toInt().coerceIn(0, sorted.lastIndex)
        return sorted[index]
    }

    private fun elapsedMs(startNs: Long): Double = (System.nanoTime() - startNs) / 1_000_000.0

    private fun formatGate(inferenceP95: Double): String = when {
        inferenceP95 <= 50.0 -> "PASS_STRONG"
        inferenceP95 <= 70.0 -> "PASS_MARGINAL"
        else -> "FAIL_CONFIG_D"
    }

    private fun readThermalStatus(): String {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            return "UNKNOWN"
        }
        return mapThermalStatus(powerManager?.currentThermalStatus)
    }

    private fun mapThermalStatus(thermalStatus: Int?): String {
        return when (thermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> "NONE"
            PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT"
            PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE"
            PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
            PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN"
            else -> "UNKNOWN"
        }
    }

    private object NoOpPerFramePerfLogger : PerFramePerfLogger {
        override val filePath: String = ""

        override fun append(row: PerFramePerfRow) = Unit

        override fun close() = Unit
    }

    companion object {
        private const val DEFAULT_MODEL_ASSET_PATH = "yolo11n_640_5-class_04-01-2026_saved_model/yolo11n_640_5-class_04-01-2026_float16.tflite"
        private const val QNN_MODEL_ASSET_PATH = "spike_qai_yolo11n_640_5-class_04-28-2026_int8.tflite"
        private const val POSE_MODEL_ASSET_PATH = "pose_landmarks_detector.tflite"
        private const val POSE_OUTPUT_FOLDER_NAME = "pose_validation"
        private const val POSE_RUN_TAG = "CVPoseDay5"
        private const val VISIBILITY_THRESHOLD = 0.6f
        private const val PER_FRAME_MODE = "sequential_yolo_pose"
        private const val POSE_VALIDATION_CSV_HEADER =
            "timestamp,file,yolo_preprocess_ms,yolo_inference_ms,yolo_nms_ms,pose_crop_ms,pose_preprocess_ms,pose_inference_ms,pose_postprocess_ms,frame_total_ms,overlay_write_ms,pose_presence,visible_joints_33,decoded_landmarks_39,status,error"
        private val NON_FILE_STEM_CHARS_REGEX = Regex("[^A-Za-z0-9._-]")

        @Volatile
        internal var testOverrides: TestOverrides? = null
    }

    internal data class TestOverrides(
        val frameProcessor: FrameProcessorGateway? = null,
        val performanceLogger: PerformanceLogger? = null,
        val trackingLogger: TrackingLogger? = null,
        val perFramePerfLogger: PerFramePerfLogger? = null,
        val modelPaths: List<String>? = null,
        val initialModelPath: String? = null,
        val gpuProbeResult: GpuProbeResult? = null,
        val qnnProbeResult: QnnProbeResult? = null,
        val nnApiProbeResult: String? = null,
        val poseOutputRootProvider: (() -> File?)? = null
    )

    private data class PoseValidationSource(
        val uri: Uri,
        val displayName: String
    )

    private data class PoseValidationInput(
        val source: PoseValidationSource,
        val bitmap: Bitmap
    )
}
