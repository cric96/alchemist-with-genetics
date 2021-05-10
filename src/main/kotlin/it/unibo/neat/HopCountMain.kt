package it.unibo.neat

import kotlin.system.exitProcess

fun main() {
    val stopWhen = 0.001
    val inputCount = 2
    val outputCount = 1
    val populationSize = 500
    val simulationFile = "hop-count.yml"
    val alchemistScore = MinCountCalculation(simulationFile, 5.0)
    val net = NeatAlgorithm.returnBest(alchemistScore, inputCount, outputCount, populationSize) { it.error > stopWhen }
    exitProcess(0)
}
