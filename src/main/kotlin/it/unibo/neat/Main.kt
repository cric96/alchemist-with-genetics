package it.unibo.neat

import org.encog.ml.MLMethod
import org.encog.ml.MLRegression
import org.encog.ml.data.basic.BasicMLData
import org.encog.neural.neat.NEATPopulation
import org.encog.neural.neat.NEATUtil
import java.util.stream.Stream
import kotlin.streams.toList

fun main() {
    val stopWhen = 0.001
    val inputCount = 2
    val outputCount = 1
    val populationSize = 50
    val simulationFile = "hop-count.yml"
    val pop = NEATPopulation(inputCount, outputCount, populationSize)
    pop.reset()
    val alchemistScore = AlchemistScoreCalculation(simulationFile)
    val evolutionLoop = Stream.iterate(NEATUtil.constructNEATTrainer(pop, alchemistScore)) { train -> train.iteration(); train }
    val train = evolutionLoop.takeWhile { it.error > stopWhen }
            .peek { println("Epoch #" + it.iteration + " Error:" + it.error + ", Species:" + pop.species.size) }
            .toList()
            .last()
    println(train.bestGenome.score)
    val net : MLMethod = train.codec.decode(train.bestGenome)
    when(net) {
        is MLRegression -> {
            println(net.compute(BasicMLData(doubleArrayOf(1.0, 1.0))))
            println(net.compute(BasicMLData(doubleArrayOf(0.0, 0.0))))
            println(net.compute(BasicMLData(doubleArrayOf(1.0, 0.0))))
            println(net.compute(BasicMLData(doubleArrayOf(0.0, 1.0))))
        }
        else -> throw IllegalArgumentException("not reachable..")
    }
    System.exit(0)
}
