val scafiVersion = "0.3.3"
val scalaVersionMajor = "2.13"
val scalaVersionMinor = ".2"
val jeneticsVersion = "6.2.0"
val deepLearningVersion = "1.0.0-M1"

plugins {
    application
    scala
    kotlin("jvm") version "1.5.0"
}

repositories {
    mavenCentral()
}
/*
 * Only required if you plan to use Protelis, remove otherwise
 */
sourceSets {
    main {
        java.srcDirs( "src/main/kotlin", "src/main/scala")
        resources {
            srcDir("src/main/yaml")
        }
    }
}
dependencies {
    // Scala Dependencies
    implementation("org.scala-lang:scala-library:$scalaVersionMajor$scalaVersionMinor")
    // The version of Alchemist can be controlled by changing the version.properties file
    implementation("it.unibo.alchemist:alchemist:_")
    implementation("it.unibo.alchemist:alchemist-incarnation-scafi:_")
    implementation("it.unibo.alchemist:alchemist-swingui:_")
    implementation("org.encog:encog-core:3.3.0")
    // ScaFi version
    implementation("it.unibo.scafi:scafi-core_$scalaVersionMajor:$scafiVersion")
    implementation("it.unibo.scafi:scafi-simulator_$scalaVersionMajor:$scafiVersion")
    implementation("it.unibo.scafi:scafi-simulator-gui_$scalaVersionMajor:$scafiVersion") {
        isTransitive = false
    }
    // Jenetics
    implementation("io.jenetics", "jenetics", jeneticsVersion)
    implementation("io.jenetics", "jenetics.prog", jeneticsVersion)
    implementation("io.jenetics", "jenetics.xml", jeneticsVersion)
    implementation("io.jenetics:prngine:1.1.0")
    // Deep Learning4J
    implementation("org.deeplearning4j", "deeplearning4j-core", deepLearningVersion)
    implementation("org.nd4j", "nd4j-native-platform", deepLearningVersion)
}

val batch: String by project
val maxTime: String by project

val alchemistGroup = "Run Alchemist"
/*
 * This task is used to run all experiments in sequence
 */
val runAll by tasks.register<DefaultTask>("runAll") {
    group = alchemistGroup
    description = "Launches all simulations"
}
/*
 * Scan the folder with the simulation files, and create a task for each one of them.
 */
File(rootProject.rootDir.path + "/src/main/yaml").listFiles()
    .filter { it.extension == "yml" } // pick all yml files in src/main/yaml
    .sortedBy { it.nameWithoutExtension } // sort them, we like reproducibility
    .forEach {
        // one simulation file -> one gradle task
        val task by tasks.register<JavaExec>("run${it.nameWithoutExtension.capitalize()}") {
            group = alchemistGroup // This is for better organization when running ./gradlew tasks
            description = "Launches simulation ${it.nameWithoutExtension}" // Just documentation
            main = "it.unibo.alchemist.Alchemist" // The class to launch
            classpath = sourceSets["main"].runtimeClasspath // The classpath to use
            // In case our simulation produces data, we write it in the following folder:
            val exportsDir = File("${projectDir.path}/build/exports/${it.nameWithoutExtension}")
            doFirst {
                // this is not executed upfront, but only when the task is actually launched
                // If the export folder doesn not exist, create it and its parents if needed
                if (!exportsDir.exists()) {
                    exportsDir.mkdirs()
                }
            }
            // These are the program arguments
            args("-y", it.absolutePath, "-e", "$exportsDir/${it.nameWithoutExtension}-${System.currentTimeMillis()}")
            if (System.getenv("CI") == "true" || batch == "true") {
                // If it is running in a Continuous Integration environment, use the "headless" mode of the simulator
                // Namely, force the simulator not to use graphical output.
                args("-hl", "-t", maxTime)
            } else {
                // A graphics environment should be available, so load the effects for the UI from the "effects" folder
                // Effects are expected to be named after the simulation file
                args("-g", "effects/${it.nameWithoutExtension}.aes")
            }
            // This tells gradle that this task may modify the content of the export directory
            outputs.dir(exportsDir)
        }
        // task.dependsOn(classpathJar) // Uncomment to switch to jar-based classpath resolution
        runAll.dependsOn(task)
    }

tasks.register("hopCountRun", JavaExec::class.java) {
    classpath = sourceSets.main.get().runtimeClasspath
    main = "it.unibo.neat.HopCountMainKt"
}