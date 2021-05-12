package it.unibo.neat

import org.encog.ml.MLRegression
import org.encog.ml.data.basic.BasicMLData
import java.io.FileOutputStream
import java.io.ObjectOutputStream
import kotlin.system.exitProcess


fun main() {
    val stopWhen = 1
    val inputCount = 2
    val outputCount = 1
    val populationSize = 100
    val howMany = 16.0 //helps to normalize nodes..
    val simulationFile = "hop-count.yml"
    val alchemistScore = MinCountCalculation(simulationFile, 10.0)
    val population = NeatAlgorithm.exec(alchemistScore, inputCount, outputCount, populationSize) { it.error > stopWhen }
    val best = population.codec.decode(population.bestGenome)
    val net = when(best) {
        is MLRegression -> best
        else -> throw IllegalArgumentException("not reachable..")
    }

    println(net.compute(BasicMLData(doubleArrayOf(0.0, 1.0))).getData(0) * howMany)
    for (d in (1..howMany.toInt()).map { it / howMany }) {
        println("$d = " + net.compute(BasicMLData(doubleArrayOf(d, 0.0))).getData(0) * howMany)
    }
    val fileOut = FileOutputStream("bestgene")
    val objectOut = ObjectOutputStream(fileOut)
    objectOut.writeObject(best)
    objectOut.close()
    println(alchemistScore.calculateScore(net))
    exitProcess(0)
}
