package it.unibo.neat

import it.unibo.alchemist.loader.LoadAlchemist
import it.unibo.alchemist.model.implementations.times.DoubleTime
import it.unibo.alchemist.model.interfaces.Position
import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.encog.ml.CalculateScore
import org.encog.ml.MLMethod
import org.encog.ml.MLRegression
import org.encog.neural.neat.NEATPopulation
import org.encog.neural.neat.NEATUtil
import java.util.stream.Stream
import kotlin.streams.toList
import it.unibo.alchemist.core.implementations.Engine as Engine1
import org.apache.log4j.varia.NullAppender




class AlchemistScoreCalculation() : CalculateScore {
    val alchemist = LoadAlchemist.from(ClassLoader.getSystemClassLoader().getResource("simulation.yml")!!)
    override fun calculateScore(method: MLMethod?): Double {
        val simulation = Engine1(alchemist.getDefault<Any, Position<*>>().environment, DoubleTime(5.0))
        when(method) {
           is MLRegression -> {
               simulation.play()
               simulation.run()
               return 10.0
           }
           else -> throw IllegalArgumentException("Alchemist accept o")
        }
    }

    override fun shouldMinimize(): Boolean {
        return true
    }

    override fun requireSingleThreaded(): Boolean {
        return false //for now..
    }

}

fun main() {
    val stopWhen = 0.0001
    val pop = NEATPopulation(2, 1, 50)
    pop.reset()
    val alchemistScore = AlchemistScoreCalculation()
    val evolutionLoop = Stream.iterate(NEATUtil.constructNEATTrainer(pop, alchemistScore)) { train -> train.iteration(); train}
    val rightNetwork = evolutionLoop.takeWhile { it.error > stopWhen }
            .peek { println("Epoch #" + it.iteration + " Error:" + it.error + ", Species:" + pop.species.size) }
            .toList()
            .last()
    println(rightNetwork.bestGenome.score)

}
