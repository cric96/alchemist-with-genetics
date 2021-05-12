package it.unibo.neat

import org.encog.ml.CalculateScore
import org.encog.ml.MLRegression
import org.encog.ml.ea.train.basic.TrainEA
import org.encog.neural.neat.NEATPopulation
import org.encog.neural.neat.NEATUtil
import java.util.stream.Stream
import kotlin.streams.toList
import kotlin.system.measureTimeMillis

object NeatAlgorithm {
    fun exec(calculation : CalculateScore, input : Int, output : Int, populationSize : Int, stopWhen : (TrainEA) -> Boolean) : TrainEA {
        val pop = NEATPopulation(input, output, populationSize)
        pop.reset()
        val evolutionLoop = Stream.iterate(NEATUtil.constructNEATTrainer(pop, calculation)) { train ->
            val time = measureTimeMillis {  train.iteration() }
            println("Time $time")
            train
        }
        val train = evolutionLoop.takeWhile(stopWhen)
            .peek { println("Epoch #" + it.iteration + " Error:" + it.error + ", Species:" + pop.species.size) }
            .toList()
            .last()
        return train
    }

    fun returnBest(calculation : CalculateScore, input : Int, output : Int, populationSize : Int, stopWhen : (TrainEA) -> Boolean) : MLRegression {
        val genetics : TrainEA = exec(calculation, input, output, populationSize, stopWhen)
        when(val best = genetics.codec.decode(genetics.bestGenome)) {
            is MLRegression -> return best
            else -> throw IllegalArgumentException("not reachable..")
        }
    }
}