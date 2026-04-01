package com.courtvision.spike.camera

import android.Manifest
import android.content.Context
import android.hardware.display.DisplayManager
import android.content.pm.PackageManager
import android.graphics.Paint
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
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
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
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

    Box(modifier = Modifier.fillMaxSize()) {
        CameraPreview(
            viewModel = viewModel,
            modifier = Modifier.fillMaxSize()
        )
        DetectionOverlay(
            detectionFrame = uiState.detectionFrame,
            modifier = Modifier.fillMaxSize()
        )
        MetricsOverlay(
            uiState = uiState,
            onModeSelect = viewModel::setInferenceMode,
            onModelSelect = viewModel::setModel,
            onConfirmModel = viewModel::confirmModel,
            onRestartSession = viewModel::restartSession
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
    val displayManager = remember {
        context.getSystemService(Context.DISPLAY_SERVICE) as DisplayManager
    }
    var displayListenerRef by remember { mutableStateOf<DisplayManager.DisplayListener?>(null) }

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
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )

            // Keep ImageAnalysis.targetRotation in sync with display rotation
            // so CameraX reports the correct rotationDegrees on each frame.
            val listener = object : DisplayManager.DisplayListener {
                override fun onDisplayChanged(displayId: Int) {
                    val display = displayManager.getDisplay(displayId) ?: return
                    preview.targetRotation = display.rotation
                    analysis.targetRotation = display.rotation
                }
                override fun onDisplayAdded(displayId: Int) {}
                override fun onDisplayRemoved(displayId: Int) {}
            }
            displayManager.registerDisplayListener(listener, null)
            displayListenerRef = listener

            viewModel.onCameraStarted()
        } catch (error: Throwable) {
            viewModel.onCameraError("Camera bind failed: ${error.message ?: "unknown error"}")
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            displayListenerRef?.let { displayManager.unregisterDisplayListener(it) }
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

    Canvas(modifier = modifier) {
        val canvasW = size.width
        val canvasH = size.height

        // Effective source dimensions after rotation (sensor reports landscape W×H).
        val rotation = detectionFrame.rotationDegrees
        val effectiveSrcW: Float
        val effectiveSrcH: Float
        if (rotation == 90 || rotation == 270) {
            effectiveSrcW = detectionFrame.sourceHeight.toFloat()
            effectiveSrcH = detectionFrame.sourceWidth.toFloat()
        } else {
            effectiveSrcW = detectionFrame.sourceWidth.toFloat()
            effectiveSrcH = detectionFrame.sourceHeight.toFloat()
        }

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
                0 -> Color(0xFFFF6F00)  // ball → orange
                1 -> Color(0xFF43A047)  // made → green
                2 -> Color(0xFF1E88E5)  // person → blue
                3 -> Color(0xFFFFD600)  // rim → yellow
                4 -> Color(0xFFAB47BC)  // shoot → purple
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
    }
}

@Composable
private fun MetricsOverlay(
    uiState: CameraUiState,
    onModeSelect: (InferenceMode) -> Unit,
    onModelSelect: (String) -> Unit,
    onConfirmModel: () -> Unit,
    onRestartSession: () -> Unit
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
private fun InferenceModeSelector(
    selectedMode: InferenceMode,
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
    }
}

@Composable
private fun ModeButton(
    label: String,
    selected: Boolean,
    onClick: () -> Unit
) {
    val colors = if (selected) {
        CardDefaults.cardColors(containerColor = Color(0xFF1976D2))
    } else {
        CardDefaults.cardColors(containerColor = Color(0x80212121))
    }

    Card(colors = colors) {
        Button(
            onClick = onClick,
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
