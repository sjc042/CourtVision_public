package com.courtvision.spike.camera

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
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
        MetricsOverlay(
            uiState = uiState,
            onDelaySelect = viewModel::setSimulatedDelayMs
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

            viewModel.onCameraStarted()
        } catch (error: Throwable) {
            viewModel.onCameraError("Camera bind failed: ${error.message ?: "unknown error"}")
        }
    }

    DisposableEffect(Unit) {
        onDispose {
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
private fun MetricsOverlay(
    uiState: CameraUiState,
    onDelaySelect: (Long) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color(0x66000000))
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
            style = MaterialTheme.typography.bodySmall
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
            "Dropped: ${uiState.stats.droppedFrames} | Queue: ${uiState.stats.queueDepth}",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall
        )
        Text(
            "CSV: ${uiState.logFilePath}",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall
        )
        uiState.lastError?.let {
            Text("Error: $it", color = Color(0xFFFFD54F), style = MaterialTheme.typography.bodySmall)
        }
        DelaySelector(
            selectedDelayMs = uiState.simulatedDelayMs,
            onDelaySelect = onDelaySelect
        )
    }
}

@Composable
private fun DelaySelector(
    selectedDelayMs: Long,
    onDelaySelect: (Long) -> Unit
) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        DelayButton(label = "0ms", selected = selectedDelayMs == 0L, onClick = { onDelaySelect(0) })
        DelayButton(label = "10ms", selected = selectedDelayMs == 10L, onClick = { onDelaySelect(10) })
        DelayButton(label = "20ms", selected = selectedDelayMs == 20L, onClick = { onDelaySelect(20) })
    }
}

@Composable
private fun DelayButton(
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
