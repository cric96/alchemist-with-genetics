package it.unibo.neat

import org.encog.ml.MLRegression
import org.encog.ml.data.basic.BasicMLData
import kotlin.system.exitProcess

fun main() {
    val stopWhen = 0.001
    val inputCount = 2
    val outputCount = 1
    val populationSize = 100
    val simulationFile = "simulation.yml"
    val alchemistScore = AndScoreCalculation(simulationFile)
    val net = NeatAlgorithm.returnBest(alchemistScore, inputCount, outputCount, populationSize) { it.error > stopWhen }
    println(net.compute(BasicMLData(doubleArrayOf(1.0, 1.0))))
    println(net.compute(BasicMLData(doubleArrayOf(0.0, 0.0))))
    println(net.compute(BasicMLData(doubleArrayOf(1.0, 0.0))))
    println(net.compute(BasicMLData(doubleArrayOf(0.0, 1.0))))
    exitProcess(0)
}
