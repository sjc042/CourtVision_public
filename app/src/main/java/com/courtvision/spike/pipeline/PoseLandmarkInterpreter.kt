package com.courtvision.spike.pipeline

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import kotlin.math.exp
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage

class PoseLandmarkInterpreter(
    modelBuffer: MappedByteBuffer,
    useGpu: Boolean = true
) : PoseInferenceEngine {

    data class OutputTensorIndices(
        val image: Int,
        val presence: Int,
        val world: Int
    )

    private val delegate: GpuDelegate?
    private val interpreter: Interpreter
    private val outputIndices: OutputTensorIndices

    private val imageOutput = Array(1) { FloatArray(PoseTensorContract.IMAGE_OUTPUT_SIZE) }
    private val presenceOutput = Array(1) { FloatArray(1) }
    private val worldOutput = Array(1) { FloatArray(PoseTensorContract.WORLD_OUTPUT_SIZE) }
    private val inferenceInputs = arrayOf<Any>(ByteBuffer.allocate(0))
    private val inferenceOutputs: MutableMap<Int, Any> = HashMap(3)

    private val preprocessor = ImageProcessor.Builder()
        .add(NormalizeOp(0f, 255f))
        .build()
    private val tensorImage = TensorImage(DataType.FLOAT32)

    init {
        val runtimeConfig = resolveRuntimeConfig(useGpu)
        val options = Interpreter.Options()
        val localDelegate = if (runtimeConfig.useGpu) {
            buildSustainedSpeedGpuDelegate().also { options.addDelegate(it) }
        } else {
            options.setNumThreads(runtimeConfig.cpuThreads)
            null
        }
        delegate = localDelegate
        interpreter = Interpreter(modelBuffer, options)
        interpreter.allocateTensors()
        validateInputTensor(interpreter.getInputTensor(0).shape(), interpreter.getInputTensor(0).dataType())
        outputIndices = resolveOutputIndices(
            outputShapes = (0 until interpreter.outputTensorCount).map {
                interpreter.getOutputTensor(it).shape()
            }
        )
        validateOutputTensorType(outputIndices.image, interpreter.getOutputTensor(outputIndices.image).dataType())
        validateOutputTensorType(outputIndices.presence, interpreter.getOutputTensor(outputIndices.presence).dataType())
        validateOutputTensorType(outputIndices.world, interpreter.getOutputTensor(outputIndices.world).dataType())
        inferenceOutputs[outputIndices.image] = imageOutput
        inferenceOutputs[outputIndices.presence] = presenceOutput
        inferenceOutputs[outputIndices.world] = worldOutput
    }

    override fun infer(cropBitmap: Bitmap): PoseResult {
        require(
            cropBitmap.width == PoseTensorContract.INPUT_SIZE &&
                cropBitmap.height == PoseTensorContract.INPUT_SIZE
        ) {
            "Pose input must be ${PoseTensorContract.INPUT_SIZE}x${PoseTensorContract.INPUT_SIZE}, " +
                "was ${cropBitmap.width}x${cropBitmap.height}"
        }
        val totalStartNs = System.nanoTime()
        val preprocessStartNs = totalStartNs

        tensorImage.load(cropBitmap)
        val processedTensor = preprocessor.process(tensorImage)
        processedTensor.buffer.rewind()

        val preprocessMs = elapsedMs(preprocessStartNs)
        val inferenceStartNs = System.nanoTime()

        inferenceInputs[0] = processedTensor.buffer
        interpreter.runForMultipleInputsOutputs(inferenceInputs, inferenceOutputs)

        val inferenceMs = elapsedMs(inferenceStartNs)
        val postprocessStartNs = System.nanoTime()

        val imageLandmarks39 = decodeImageLandmarks(imageOutput[0])
        val worldLandmarks39 = decodeWorldLandmarks(worldOutput[0])
        val presenceLogit = presenceOutput[0][0]
        val posePresence = sigmoid(presenceLogit)

        val postprocessMs = elapsedMs(postprocessStartNs)
        val totalMs = elapsedMs(totalStartNs)

        return PoseResult(
            imageLandmarks39 = imageLandmarks39,
            worldLandmarks39 = worldLandmarks39,
            imageLandmarks33 = imageLandmarks39.take(PoseTensorContract.LANDMARKS_CANONICAL),
            worldLandmarks33 = worldLandmarks39.take(PoseTensorContract.LANDMARKS_CANONICAL),
            posePresenceLogit = presenceLogit,
            posePresence = posePresence,
            latency = PoseStageLatency(
                preprocessMs = preprocessMs,
                inferenceMs = inferenceMs,
                postprocessMs = postprocessMs,
                totalMs = totalMs
            )
        )
    }

    override fun close() {
        interpreter.close()
        delegate?.close()
    }

    companion object {
        internal data class RuntimeConfig(
            val useGpu: Boolean,
            val cpuThreads: Int
        )

        internal fun resolveRuntimeConfig(useGpu: Boolean): RuntimeConfig =
            if (useGpu) {
                RuntimeConfig(useGpu = true, cpuThreads = 0)
            } else {
                RuntimeConfig(useGpu = false, cpuThreads = 4)
            }

        internal fun resolveOutputIndices(outputShapes: List<IntArray>): OutputTensorIndices {
            var imageIndex = -1
            var presenceIndex = -1
            var worldIndex = -1

            outputShapes.forEachIndexed { index, shape ->
                when {
                    shape.contentEquals(intArrayOf(1, PoseTensorContract.IMAGE_OUTPUT_SIZE)) -> imageIndex = index
                    shape.contentEquals(intArrayOf(1, 1)) -> presenceIndex = index
                    shape.contentEquals(intArrayOf(1, PoseTensorContract.WORLD_OUTPUT_SIZE)) -> worldIndex = index
                }
            }

            require(imageIndex >= 0) {
                "Missing image landmark output shape [1, ${PoseTensorContract.IMAGE_OUTPUT_SIZE}]"
            }
            require(presenceIndex >= 0) {
                "Missing pose presence output shape [1, 1]"
            }
            require(worldIndex >= 0) {
                "Missing world landmark output shape [1, ${PoseTensorContract.WORLD_OUTPUT_SIZE}]"
            }

            return OutputTensorIndices(
                image = imageIndex,
                presence = presenceIndex,
                world = worldIndex
            )
        }

        internal fun validateInputTensor(inputShape: IntArray, inputType: DataType) {
            require(
                inputShape.contentEquals(
                    intArrayOf(
                        1,
                        PoseTensorContract.INPUT_SIZE,
                        PoseTensorContract.INPUT_SIZE,
                        3
                    )
                )
            ) {
                "Unexpected pose input shape: ${inputShape.contentToString()}"
            }
            require(inputType == DataType.FLOAT32) {
                "Unexpected pose input dtype: $inputType"
            }
        }

        private fun validateOutputTensorType(outputIndex: Int, outputType: DataType) {
            require(outputType == DataType.FLOAT32) {
                "Unexpected pose output dtype at index $outputIndex: $outputType"
            }
        }

        internal fun decodeImageLandmarks(raw: FloatArray): List<PoseImageLandmark> {
            require(raw.size == PoseTensorContract.IMAGE_OUTPUT_SIZE) {
                "Unexpected image output size: ${raw.size}"
            }
            val out = ArrayList<PoseImageLandmark>(PoseTensorContract.LANDMARKS_TOTAL)
            var offset = 0
            repeat(PoseTensorContract.LANDMARKS_TOTAL) {
                val x = raw[offset]
                val y = raw[offset + 1]
                val z = raw[offset + 2]
                val visibilityLogit = raw[offset + 3]
                val presenceLogit = raw[offset + 4]
                out.add(
                    PoseImageLandmark(
                        xPx = x,
                        yPx = y,
                        zPx = z,
                        visibilityLogit = visibilityLogit,
                        presenceLogit = presenceLogit,
                        visibility = sigmoid(visibilityLogit),
                        presence = sigmoid(presenceLogit)
                    )
                )
                offset += PoseTensorContract.IMAGE_FIELDS
            }
            return out
        }

        internal fun decodeWorldLandmarks(raw: FloatArray): List<PoseWorldLandmark> {
            require(raw.size == PoseTensorContract.WORLD_OUTPUT_SIZE) {
                "Unexpected world output size: ${raw.size}"
            }
            val out = ArrayList<PoseWorldLandmark>(PoseTensorContract.LANDMARKS_TOTAL)
            var offset = 0
            repeat(PoseTensorContract.LANDMARKS_TOTAL) {
                out.add(
                    PoseWorldLandmark(
                        x = raw[offset],
                        y = raw[offset + 1],
                        z = raw[offset + 2]
                    )
                )
                offset += PoseTensorContract.WORLD_FIELDS
            }
            return out
        }

        internal fun sigmoid(value: Float): Float = (1f / (1f + exp(-value)))

        private fun elapsedMs(startNs: Long): Double = (System.nanoTime() - startNs) / 1_000_000.0
    }
}
