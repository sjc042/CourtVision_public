package com.courtvision.spike.camera

import android.Manifest
import android.content.Context
import android.view.OrientationEventListener
import android.view.Surface
import android.content.pm.PackageManager
import android.os.SystemClock
import android.graphics.Paint
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.PickVisualMediaRequest
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.courtvision.spike.pipeline.DetectionFrame
import com.courtvision.spike.pipeline.InferenceMode
import com.courtvision.spike.pipeline.LivePoseOverlay
import com.courtvision.spike.pipeline.PoseTensorContract
import com.courtvision.spike.pipeline.RotationStallState
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import java.util.Locale
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine

@Composable
fun CameraScreen(
    viewModel: CameraViewModel = viewModel()
) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    var hasPermission by remember { mutableStateOf(context.hasCameraPermission()) }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasPermission = granted
        viewModel.onCameraPermissionResult(granted)
    }
    val poseValidationPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickMultipleVisualMedia()
    ) { uris ->
        viewModel.startDay5PoseIsolatedValidation(uris)
    }

    LaunchedEffect(hasPermission) {
        viewModel.onCameraPermissionResult(hasPermission)
    }

    if (!hasPermission) {
        PermissionRequired(
            onRequestPermission = {
                permissionLauncher.launch(Manifest.permission.CAMERA)
            }
        )
        return
    }

    var detectionOverlayLagMs by remember { mutableStateOf(0.0) }
    var poseOverlayLagMs by remember { mutableStateOf(0.0) }

    LaunchedEffect(uiState.detectionFrame.emitElapsedRealtimeNanos) {
        val emitNs = uiState.detectionFrame.emitElapsedRealtimeNanos
        if (emitNs > 0L) {
            withFrameNanos { frameNs ->
                detectionOverlayLagMs = (frameNs - emitNs) / 1_000_000.0
            }
        } else {
            detectionOverlayLagMs = 0.0
        }
    }

    LaunchedEffect(uiState.poseOverlay?.emitElapsedRealtimeNanos) {
        val emitNs = uiState.poseOverlay?.emitElapsedRealtimeNanos ?: 0L
        if (emitNs > 0L) {
            withFrameNanos { frameNs ->
                poseOverlayLagMs = (frameNs - emitNs) / 1_000_000.0
            }
        } else {
            poseOverlayLagMs = 0.0
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        CameraPreview(
            viewModel = viewModel,
            modifier = Modifier.fillMaxSize()
        )
        DetectionOverlay(
            detectionFrame = uiState.detectionFrame,
            modifier = Modifier.fillMaxSize()
        )
        PoseOverlay(
            overlay = uiState.poseOverlay,
            sourceWidth = uiState.detectionFrame.sourceWidth,
            sourceHeight = uiState.detectionFrame.sourceHeight,
            modifier = Modifier.fillMaxSize()
        )
        MetricsOverlay(
            uiState = uiState,
            detectionOverlayLagMs = detectionOverlayLagMs,
            poseOverlayLagMs = poseOverlayLagMs,
            onModeSelect = viewModel::setInferenceMode,
            onTrackerMissFramesChanged = viewModel::setTrackerMaxMissFrames,
            onTrackerNoiseChanged = viewModel::setTrackerNoise,
            onModelSelect = viewModel::setModel,
            onConfirmModel = viewModel::confirmModel,
            onRestartSession = viewModel::restartSession,
            onRunPoseValidation = {
                poseValidationPickerLauncher.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                )
            }
        )
    }
}

@Composable
private fun CameraPreview(
    viewModel: CameraViewModel,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    var orientationListenerRef by remember { mutableStateOf<OrientationEventListener?>(null) }
    var reconcileJobRef by remember { mutableStateOf<Job?>(null) }

    LaunchedEffect(previewView, lifecycleOwner) {
        try {
            val cameraProvider = context.awaitCameraProvider()
            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetResolution(Size(1280, 720))
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()

            analysis.setAnalyzer(cameraExecutor, viewModel.imageAnalyzer())

            cameraProvider.unbindAll()
            val camera = cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )

            val expectedTargetRotation = AtomicInteger(
                normalizeSurfaceRotation(previewView.display?.rotation ?: Surface.ROTATION_0)
            )
            var applySeq = 0L
            var lastRecoveryRebindElapsedMs = -1L

            fun applyTargetRotation(targetRotation: Int, source: String) {
                val normalizedTarget = normalizeSurfaceRotation(targetRotation)
                val callbackIndex = ++applySeq
                val eventUptimeMs = SystemClock.uptimeMillis()
                val callbackStartNs = SystemClock.elapsedRealtimeNanos()
                val callbackThread = Thread.currentThread().name
                val previousExpectedTarget = expectedTargetRotation.getAndSet(normalizedTarget)
                val analysisBefore = analysis.targetRotation
                val previewBefore = preview.targetRotation
                preview.targetRotation = normalizedTarget
                analysis.targetRotation = normalizedTarget
                val expectedFrameRotationDegrees =
                    camera.cameraInfo.getSensorRotationDegrees(normalizedTarget)

                viewModel.updateExpectedRotation(
                    expectedTargetRotation = normalizedTarget,
                    expectedFrameRotationDegrees = expectedFrameRotationDegrees,
                    source = source
                )

                val callbackDurationMs =
                    (SystemClock.elapsedRealtimeNanos() - callbackStartNs) / 1_000_000.0
                if (ROTATION_DEBUG_LOGS) {
                    android.util.Log.i(
                        CV_ROTATION_TAG,
                        "[UI][TARGET_APPLY] seq=$callbackIndex source=$source eventUptimeMs=$eventUptimeMs thread=$callbackThread " +
                            "expectedTargetBefore=$previousExpectedTarget expectedTargetAfter=$normalizedTarget " +
                            "expectedRaw=$expectedFrameRotationDegrees analysisTargetBefore=$analysisBefore analysisTargetAfter=${analysis.targetRotation} " +
                            "previewTargetBefore=$previewBefore previewTargetAfter=${preview.targetRotation} " +
                            "callbackDurationMs=${"%.3f".format(callbackDurationMs)}"
                    )
                }
            }

            val orientationListener = object : OrientationEventListener(context) {
                override fun onOrientationChanged(orientation: Int) {
                    if (orientation == ORIENTATION_UNKNOWN) return
                    val newRotation = orientationToSurfaceRotation(orientation)
                    if (newRotation != expectedTargetRotation.get()) {
                        if (ROTATION_DEBUG_LOGS) {
                            android.util.Log.i(
                                CV_ROTATION_TAG,
                                "[UI][OEL_ROT_CHANGE] orientation=$orientation targetRotation=$newRotation expectedTarget=${expectedTargetRotation.get()}"
                            )
                        }
                        applyTargetRotation(newRotation, source = "orientation_event")
                    }
                }
            }
            orientationListener.enable()
            orientationListenerRef = orientationListener

            applyTargetRotation(expectedTargetRotation.get(), source = "initial_bind")

            val reconcileJob = launch {
                while (isActive) {
                    delay(RECONCILE_TICK_MS)
                    val telemetry = viewModel.rotationTelemetrySnapshot()
                    val stallState = telemetry.stallState
                    if (stallState != RotationStallState.RECONCILE &&
                        stallState != RotationStallState.RECOVERY_REQUESTED
                    ) {
                        continue
                    }

                    val expectedTarget = expectedTargetRotation.get()
                    android.util.Log.w(
                        CV_ROTATION_TAG,
                        "[UI][RECONCILE_APPLY] expectedTarget=$expectedTarget expectedRaw=${telemetry.expectedFrameRotationDegrees} " +
                            "frameRaw=${telemetry.frameRotationDegrees} mismatchMs=${telemetry.rotationMismatchMs} stallState=$stallState dropped=${telemetry.droppedFramesSnapshot}"
                    )
                    applyTargetRotation(expectedTarget, source = "reconcile_tick")

                    val nowMs = SystemClock.elapsedRealtime()
                    val shouldRebind = stallState == RotationStallState.RECOVERY_REQUESTED &&
                        (lastRecoveryRebindElapsedMs < 0L ||
                            nowMs - lastRecoveryRebindElapsedMs >= RECOVERY_REBIND_COOLDOWN_MS)
                    if (!shouldRebind) continue

                    try {
                        android.util.Log.w(
                            CV_ROTATION_TAG,
                            "[UI][ROT_RECOVERY_REBIND] expectedTarget=$expectedTarget expectedRaw=${telemetry.expectedFrameRotationDegrees} " +
                                "frameRaw=${telemetry.frameRotationDegrees} mismatchMs=${telemetry.rotationMismatchMs}"
                        )
                        cameraProvider.unbind(analysis)
                        analysis.setAnalyzer(cameraExecutor, viewModel.imageAnalyzer())
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            analysis
                        )
                        lastRecoveryRebindElapsedMs = nowMs
                        applyTargetRotation(expectedTarget, source = "recovery_rebind")
                    } catch (error: Throwable) {
                        viewModel.onCameraError(
                            "Rotation recovery rebind failed: ${error.message ?: "unknown error"}"
                        )
                    }
                }
            }
            reconcileJobRef = reconcileJob

            viewModel.onCameraStarted()
        } catch (error: Throwable) {
            viewModel.onCameraError("Camera bind failed: ${error.message ?: "unknown error"}")
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            orientationListenerRef?.disable()
            reconcileJobRef?.cancel()
            try {
                ProcessCameraProvider.getInstance(context).get().unbindAll()
            } catch (_: Throwable) {
                // No-op during teardown.
            }
            cameraExecutor.shutdown()
            viewModel.onCameraStopped()
        }
    }

    AndroidView(
        factory = { previewView },
        modifier = modifier
    )
}

@Composable
private fun DetectionOverlay(
    detectionFrame: DetectionFrame,
    modifier: Modifier = Modifier
) {
    val textPaint = remember {
        Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 36f
            isAntiAlias = true
            style = Paint.Style.FILL
        }
    }
    val trackerTextPaint = remember {
        Paint().apply {
            color = android.graphics.Color.CYAN
            textSize = 34f
            isAntiAlias = true
            style = Paint.Style.FILL
        }
    }

    Canvas(modifier = modifier) {
        val canvasW = size.width
        val canvasH = size.height

        // FrameProcessor pre-rotates sensor frames before YOLO and pose, so sourceWidth/Height
        // are already in display orientation and rotationDegrees is always 0.
        val effectiveSrcW = detectionFrame.sourceWidth.toFloat()
        val effectiveSrcH = detectionFrame.sourceHeight.toFloat()

        // FILL_CENTER: scale image to fully cover the canvas, then center-crop overflow.
        val scale = if (effectiveSrcW > 0f && effectiveSrcH > 0f) {
            maxOf(canvasW / effectiveSrcW, canvasH / effectiveSrcH)
        } else {
            1f
        }
        val scaledW = effectiveSrcW * scale
        val scaledH = effectiveSrcH * scale
        val offsetX = (scaledW - canvasW) / 2f
        val offsetY = (scaledH - canvasH) / 2f


        detectionFrame.boxes.forEach { box ->
            val displayBox = box
            val left = displayBox.left * scaledW - offsetX
            val top = displayBox.top * scaledH - offsetY
            val right = displayBox.right * scaledW - offsetX
            val bottom = displayBox.bottom * scaledH - offsetY

            val boxColor = when (displayBox.classId) {
                0 -> Color(0xFFFF6F00) // ball
                1 -> Color(0xFF43A047) // made
                2 -> Color(0xFF1E88E5) // person
                3 -> Color(0xFFFFD600) // rim
                4 -> Color(0xFFAB47BC) // shoot
                else -> Color(0xFFFFFFFF)
            }

            drawRect(
                color = boxColor,
                topLeft = Offset(left, top),
                size = androidx.compose.ui.geometry.Size(
                    width = (right - left).coerceAtLeast(0f),
                    height = (bottom - top).coerceAtLeast(0f)
                ),
                style = Stroke(width = 3.dp.toPx())
            )

            drawContext.canvas.nativeCanvas.drawText(
                "${displayBox.label} ${"%.2f".format(displayBox.confidence)}",
                left.coerceAtLeast(8f),
                (top - 12f).coerceAtLeast(36f),
                textPaint
            )
        }

        val trackedBall = detectionFrame.trackedBall
        if (trackedBall?.isTracked == true) {
            val centerX = trackedBall.centroidX * scaledW - offsetX
            val centerY = trackedBall.centroidY * scaledH - offsetY
            val vectorSeconds = 0.15f
            val endX = (trackedBall.centroidX + trackedBall.velocityX * vectorSeconds) * scaledW - offsetX
            val endY = (trackedBall.centroidY + trackedBall.velocityY * vectorSeconds) * scaledH - offsetY

            drawCircle(
                color = Color.Cyan,
                radius = 6.dp.toPx(),
                center = Offset(centerX, centerY)
            )
            drawLine(
                color = Color.Cyan,
                start = Offset(centerX, centerY),
                end = Offset(endX, endY),
                strokeWidth = 2.dp.toPx()
            )
            drawContext.canvas.nativeCanvas.drawText(
                "T",
                (centerX + 8f).coerceAtMost(canvasW - 24f),
                (centerY - 8f).coerceAtLeast(32f),
                trackerTextPaint
            )
        }
    }
}

@Composable
private fun PoseOverlay(
    overlay: LivePoseOverlay?,
    sourceWidth: Int,
    sourceHeight: Int,
    modifier: Modifier = Modifier
) {
    if (overlay == null || sourceWidth <= 0 || sourceHeight <= 0) return

    Canvas(modifier = modifier) {
        val canvasW = size.width
        val canvasH = size.height
        val srcW = sourceWidth.toFloat()
        val srcH = sourceHeight.toFloat()

        val scale = maxOf(canvasW / srcW, canvasH / srcH)
        val scaledW = srcW * scale
        val scaledH = srcH * scale
        val offsetX = (scaledW - canvasW) / 2f
        val offsetY = (scaledH - canvasH) / 2f

        val cropRect = overlay.cropRectNormalized
        val cropW = cropRect.right - cropRect.left
        val cropH = cropRect.bottom - cropRect.top
        val inputSize = PoseTensorContract.INPUT_SIZE.toFloat()

        overlay.poseResult.imageLandmarks33.forEach { landmark ->
            val displayX = cropRect.left + (landmark.xPx / inputSize) * cropW
            val displayY = cropRect.top + (landmark.yPx / inputSize) * cropH
            val screenX = displayX * scaledW - offsetX
            val screenY = displayY * scaledH - offsetY
            val color = if (landmark.visibility > POSE_VISIBILITY_THRESHOLD) {
                Color.Green
            } else {
                Color.Red
            }

            drawCircle(
                color = color,
                radius = 4.dp.toPx(),
                center = Offset(screenX, screenY)
            )
        }
    }
}

@Composable
private fun MetricsOverlay(
    uiState: CameraUiState,
    detectionOverlayLagMs: Double = 0.0,
    poseOverlayLagMs: Double = 0.0,
    onModeSelect: (InferenceMode) -> Unit,
    onTrackerMissFramesChanged: (Int) -> Unit,
    onTrackerNoiseChanged: (Float, Float) -> Unit,
    onModelSelect: (String) -> Unit,
    onConfirmModel: () -> Unit,
    onRestartSession: () -> Unit,
    onRunPoseValidation: () -> Unit
) {
    val modelSelectionLocked = uiState.modelConfirmed

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color(0x66000000))
            .verticalScroll(rememberScrollState())
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text("Status: ${uiState.cameraStatus}", color = Color.White, style = MaterialTheme.typography.bodyMedium)
        Text(
            "GPU Probe: ${uiState.gpuProbeResult.status}",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall
        )
        Text(
            "NNAPI Probe: ${uiState.nnApiProbeResult}",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
        Text(
            "Device: ${uiState.gpuProbeResult.deviceModel} | API ${uiState.gpuProbeResult.apiLevel}",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
        uiState.gpuProbeResult.reason?.let { reason ->
            Text(
                "GPU Reason: $reason",
                color = Color(0xFFFFD54F),
                style = MaterialTheme.typography.bodySmall
            )
        }
        Text(
            "FPS: ${uiState.stats.analysisFps} | avg: ${"%.2f".format(uiState.stats.avgAnalyzeMs)}ms | p95: ${"%.2f".format(uiState.stats.p95AnalyzeMs)}ms",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall
        )
        Text(
            "Inference: ${"%.2f".format(uiState.stats.lastInferenceMs)}ms | Mode: ${uiState.stats.delegateMode}",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall
        )
        Text(
            "DET overlay: ${formatOverlayLag(detectionOverlayLagMs)}",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall
        )
        Text(
            "POSE overlay: ${formatOverlayLag(poseOverlayLagMs)}",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall
        )
        Text(
            "Model: ${shortModelName(uiState.selectedModel)}",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall
        )
        Text(
            "RAM: ${"%.1f".format(uiState.stats.ramMb)}MB | Thermal: ${uiState.stats.thermalStatus}",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall
        )
        Text(
            "Dropped: ${uiState.stats.droppedFrames} | Queue: ${uiState.stats.queueDepth}",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall
        )
        Text(
            "CSV: ${uiState.logFilePath}",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
        Text(
            "Track CSV: ${uiState.trackingLogFilePath}",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
        Text(
            "Pose Validation: ${uiState.poseValidationStatus}",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall
        )
        if (uiState.poseValidationOutputPath.isNotBlank()) {
            Text(
                "Pose Output: ${uiState.poseValidationOutputPath}",
                color = Color.White,
                style = MaterialTheme.typography.bodySmall,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
        Button(
            onClick = onRunPoseValidation,
            enabled = !uiState.poseValidationRunning && uiState.modelConfirmed
        ) {
            Text("Select Day5 Pose Images")
        }
        
        HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp), color = Color.White.copy(alpha = 0.2f))
        Text("Tracker Tuning", color = Color.Cyan, style = MaterialTheme.typography.titleSmall)

        Text(
            "Miss reset: ${uiState.trackerMaxMissFrames} frames",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall
        )
        Slider(
            value = uiState.trackerMaxMissFrames.toFloat(),
            onValueChange = { onTrackerMissFramesChanged(it.roundToInt()) },
            valueRange = 1f..30f,
            steps = 28
        )

        LogarithmicNoiseSlider(
            label = "Process Noise",
            value = uiState.trackerProcessNoise,
            onValueChange = { onTrackerNoiseChanged(it, uiState.trackerMeasurementNoise) }
        )

        LogarithmicNoiseSlider(
            label = "Measurement Noise",
            value = uiState.trackerMeasurementNoise,
            onValueChange = { onTrackerNoiseChanged(uiState.trackerProcessNoise, it) }
        )

        HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp), color = Color.White.copy(alpha = 0.2f))

        if (uiState.isSwitchingMode) {
            Text(
                "Switching delegate mode...",
                color = Color(0xFFFFD54F),
                style = MaterialTheme.typography.bodySmall,
                fontWeight = FontWeight.SemiBold
            )
        }
        uiState.lastError?.let {
            Text("Error: $it", color = Color(0xFFFFD54F), style = MaterialTheme.typography.bodySmall)
        }
        InferenceModeSelector(
            selectedMode = uiState.selectedMode,
            nnapiAvailable = uiState.nnApiAvailable,
            onModeSelect = onModeSelect
        )
        ModelSelector(
            availableModels = uiState.availableModels,
            selectedModel = uiState.selectedModel,
            modelConfirmed = uiState.modelConfirmed,
            locked = modelSelectionLocked,
            onModelSelect = onModelSelect,
            onConfirmModel = onConfirmModel,
            onRestartSession = onRestartSession
        )
    }
}

@Composable
private fun LogarithmicNoiseSlider(
    label: String,
    value: Float,
    onValueChange: (Float) -> Unit
) {
    // Noise values usually range from 1e-6 to 1e-1.
    // We'll use log10 for the slider to make it usable across decades.
    val logValue = remember(value) { log10(value.toDouble()).toFloat().coerceIn(-6f, -1f) }

    Column {
        Text(
            text = "$label: ${"%.1e".format(value)}",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall
        )
        Slider(
            value = logValue,
            onValueChange = { onValueChange(10f.pow(it)) },
            valueRange = -6f..-1f
        )
    }
}

@Composable
private fun InferenceModeSelector(
    selectedMode: InferenceMode,
    nnapiAvailable: Boolean,
    onModeSelect: (InferenceMode) -> Unit
) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        ModeButton(
            label = "CPU",
            selected = selectedMode == InferenceMode.CPU,
            onClick = { onModeSelect(InferenceMode.CPU) }
        )
        ModeButton(
            label = "GPU",
            selected = selectedMode == InferenceMode.GPU,
            onClick = { onModeSelect(InferenceMode.GPU) }
        )
        ModeButton(
            label = "NNAPI",
            selected = selectedMode == InferenceMode.NNAPI,
            enabled = nnapiAvailable,
            onClick = { onModeSelect(InferenceMode.NNAPI) }
        )
    }
}

@Composable
private fun ModeButton(
    label: String,
    selected: Boolean,
    enabled: Boolean = true,
    onClick: () -> Unit
) {
    val colors = when {
        selected -> CardDefaults.cardColors(containerColor = Color(0xFF1976D2))
        !enabled -> CardDefaults.cardColors(containerColor = Color(0x40212121))
        else -> CardDefaults.cardColors(containerColor = Color(0x80212121))
    }

    Card(colors = colors) {
        Button(
            onClick = onClick,
            enabled = enabled,
            modifier = Modifier.width(86.dp)
        ) {
            Text(label)
        }
    }
}

@Composable
private fun ModelSelector(
    availableModels: List<String>,
    selectedModel: String,
    modelConfirmed: Boolean,
    locked: Boolean,
    onModelSelect: (String) -> Unit,
    onConfirmModel: () -> Unit,
    onRestartSession: () -> Unit
) {
    Text(
        text = if (locked) {
            "Model selector locked after confirmation"
        } else {
            "Select model before starting inference"
        },
        color = Color.White,
        style = MaterialTheme.typography.bodySmall
    )
    Text(
        text = "Models found: ${availableModels.size}",
        color = Color.White,
        style = MaterialTheme.typography.bodySmall
    )

    if (!modelConfirmed) {
        Button(onClick = onConfirmModel) {
            Text("Start Inference")
        }
    } else {
        Button(onClick = onRestartSession) {
            Text("Restart Session")
        }
    }

    Row(
        modifier = Modifier.horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        availableModels.forEach { modelPath ->
            FilterChip(
                selected = modelPath == selectedModel,
                onClick = { onModelSelect(modelPath) },
                enabled = !locked,
                label = {
                    Text(shortModelName(modelPath))
                }
            )
        }
    }
}

private fun shortModelName(modelPath: String): String {
    if (modelPath.isBlank()) return "n/a"
    val fileName = modelPath.substringAfterLast('/').removeSuffix(".tflite")
    return fileName
        .replace("_float16", "-fp16")
        .replace("_float32", "-fp32")
}

private fun formatOverlayLag(lagMs: Double): String {
    return if (lagMs <= 0.0) "—" else String.format(Locale.US, "%.1f ms", lagMs)
}

@Composable
private fun PermissionRequired(
    onRequestPermission: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("Camera permission is required to run the spike.")
            Button(onClick = onRequestPermission) {
                Text("Grant camera permission")
            }
        }
    }
}

private fun Context.hasCameraPermission(): Boolean {
    return ContextCompat.checkSelfPermission(
        this,
        Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED
}

private suspend fun Context.awaitCameraProvider(): ProcessCameraProvider {
    return suspendCancellableCoroutine { continuation ->
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener(
            {
                try {
                    continuation.resume(future.get())
                } catch (error: Throwable) {
                    continuation.resumeWithException(error)
                }
            },
            ContextCompat.getMainExecutor(this)
        )
    }
}

private fun orientationToSurfaceRotation(orientationDegrees: Int): Int {
    return when (orientationDegrees) {
        in 45 until 135 -> Surface.ROTATION_270
        in 135 until 225 -> Surface.ROTATION_180
        in 225 until 315 -> Surface.ROTATION_90
        else -> Surface.ROTATION_0
    }
}

private fun normalizeSurfaceRotation(rotation: Int): Int {
    return when (rotation) {
        Surface.ROTATION_0,
        Surface.ROTATION_90,
        Surface.ROTATION_180,
        Surface.ROTATION_270 -> rotation
        else -> Surface.ROTATION_0
    }
}

private const val CV_ROTATION_TAG = "CVRotation"
private const val ROTATION_DEBUG_LOGS = false
private const val RECONCILE_TICK_MS = 1_000L
private const val RECOVERY_REBIND_COOLDOWN_MS = 5_000L
private const val POSE_VISIBILITY_THRESHOLD = 0.6f
