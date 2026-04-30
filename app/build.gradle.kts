plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("tech.apter.junit5.jupiter.robolectric-extension-gradle-plugin") version "0.9.0"
}

android {
    namespace = "com.courtvision.spike"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.courtvision.spike"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.14"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    androidResources {
        noCompress += "tflite"
    }
}

val pinnedKotlinVersionForTests = "1.9.24"
val pinnedKotlinArtifactsForTests = setOf(
    "kotlin-stdlib",
    "kotlin-stdlib-common",
    "kotlin-stdlib-jdk7",
    "kotlin-stdlib-jdk8",
    "kotlin-reflect"
)

configurations.configureEach {
    if (name.contains("Test")) {
        resolutionStrategy.eachDependency {
            if (
                requested.group == "org.jetbrains.kotlin" &&
                requested.name in pinnedKotlinArtifactsForTests
            ) {
                useVersion(pinnedKotlinVersionForTests)
                because("Align Kotlin runtime on test classpaths with Kotlin 1.9.24 compiler/toolchain.")
            }
        }
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.06.00")
    val junitBom = platform("org.junit:junit-bom:5.10.2")
    val cameraxVersion = "1.3.4"

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.3")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.3")
    implementation("androidx.activity:activity-compose:1.9.1")

    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")

    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    implementation("com.qualcomm.qti:qnn-runtime:2.40.0")
    implementation("com.qualcomm.qti:qnn-litert-delegate:2.40.0")

    implementation("com.google.android.material:material:1.12.0")

    testImplementation(junitBom)
    testImplementation("org.junit.jupiter:junit-jupiter-api")
    testImplementation("org.junit.jupiter:junit-jupiter-params")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.1")
    testImplementation("org.robolectric:robolectric:4.13")
    testImplementation("tech.apter.junit5.jupiter:robolectric-extension:0.9.0")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
    androidTestImplementation(composeBom)
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}

tasks.register("checkAll") {
    description = "Runs lint, unit tests, and instrumented tests (if device connected)"
    group = "verification"
    dependsOn("lintDebug", "testDebugUnitTest", "connectedDebugAndroidTest")
}

afterEvaluate {
    tasks.named("connectedDebugAndroidTest") {
        val adb = android.buildToolsVersion // force evaluation
        onlyIf {
            val result = providers.exec {
                commandLine(android.adbExecutable.absolutePath, "devices")
                isIgnoreExitValue = true
            }.standardOutput.asText.get()
            val hasDevice = result.lines().drop(1).any { it.contains("device") }
            if (!hasDevice) {
                logger.warn("WARNING: No device/emulator connected — skipping instrumented tests.")
            }
            hasDevice
        }
    }
}
