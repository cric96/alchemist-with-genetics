package it.unibo.neat

import org.encog.ml.MLMethod
import org.encog.ml.MLRegression
import org.encog.ml.data.basic.BasicMLData
import org.encog.neural.neat.NEATNetwork
import org.encog.neural.neat.NEATPopulation
import org.encog.neural.neat.NEATUtil
import java.lang.IllegalArgumentException
import java.util.stream.Stream
import kotlin.streams.toList

fun main() {
    val stopWhen = 0.001
    val pop = NEATPopulation(2, 1, 50)
    pop.reset()
    val alchemistScore = AlchemistScoreCalculation("simulation.yml")
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
