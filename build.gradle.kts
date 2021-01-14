/*
 * DEFAULT GRADLE BUILD FOR ALCHEMIST SIMULATOR
 */

plugins {
    application
}

repositories {
    mavenCentral()
}
/*
 * Only required if you plan to use Protelis, remove otherwise
 */
sourceSets {
    main {
        resources {
            srcDir("src/main/protelis")
        }
    }
}
dependencies {
    // The version of Alchemist can be controlled by changing the version.properties file
    implementation("it.unibo.alchemist:alchemist:_")
    implementation("it.unibo.alchemist:alchemist-incarnation-protelis:_")
    implementation("it.unibo.alchemist:alchemist-swingui:_")
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
    .filter { it.extension == "yml" }
    .sortedBy { it.nameWithoutExtension }
    .forEach {
        val task by tasks.register<JavaExec>("run${it.nameWithoutExtension.capitalize()}") {
            group = alchemistGroup
            description = "Launches simulation ${it.nameWithoutExtension}"
            main = "it.unibo.alchemist.Alchemist"
            classpath = sourceSets["main"].runtimeClasspath
            val exportsDir = File("${projectDir.path}/build/exports/${it.nameWithoutExtension}")
            doFirst {
                if (!exportsDir.exists()) {
                    exportsDir.mkdirs()
                }
            }
            args("-y", it.absolutePath, "-e", "$exportsDir/${it.nameWithoutExtension}-${System.currentTimeMillis()}")
            if (System.getenv("CI") == "true" || batch == "true") {
                args("-hl", "-t", maxTime)
            } else {
                args("-g", "effects/${it.nameWithoutExtension}.aes")
            }
            outputs.dir(exportsDir)
        }
        // task.dependsOn(classpathJar) // Uncomment to switch to jar-based cp resolution
        runAll.dependsOn(task)
    }

