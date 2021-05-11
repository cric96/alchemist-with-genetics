package it.unibo.neat

import kotlin.system.exitProcess

fun main() {
    val stopWhen = 0.1
    val inputCount = 2
    val outputCount = 1
    val populationSize = 100
    val simulationFile = "hop-count.yml"
    val alchemistScore = MinCountCalculation(simulationFile, 10.0)
    val net = NeatAlgorithm.returnBest(alchemistScore, inputCount, outputCount, populationSize) { it.error > stopWhen }
    exitProcess(0)
}
