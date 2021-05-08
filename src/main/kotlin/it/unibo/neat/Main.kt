package it.unibo.neat

import it.unibo.alchemist.loader.LoadAlchemist
import it.unibo.alchemist.loader.Loader
import it.unibo.alchemist.loader.SimulationModel
import it.unibo.alchemist.model.implementations.molecules.SimpleMolecule
import it.unibo.alchemist.model.implementations.times.DoubleTime
import it.unibo.alchemist.model.interfaces.Position
import org.encog.ml.CalculateScore
import org.encog.ml.MLMethod
import org.encog.ml.MLRegression
import org.encog.neural.neat.NEATPopulation
import org.encog.neural.neat.NEATUtil
import java.util.stream.Stream
import kotlin.streams.toList
import it.unibo.alchemist.core.implementations.Engine as Engine1

class AlchemistScoreCalculation(val maxTime : Double = 1.0) : CalculateScore {
    override fun calculateScore(method: MLMethod?): Double {
        val alchemist = LoadAlchemist.from(ClassLoader.getSystemClassLoader().getResource("simulation.yml")!!)
        val simulation = Engine1(alchemist.getDefault<Any, Position<*>>().environment, DoubleTime(maxTime))
        when(method) {
           is MLRegression -> {
               simulation.environment.nodes.forEach {
                   node -> node.setConcentration(SimpleMolecule("regressor"), method)
               }
               simulation.play()
               simulation.run()
               return 10.0
           }
           else -> throw IllegalArgumentException("Alchemist accept only ml regression")
        }
    }

    override fun shouldMinimize(): Boolean { return true }

    //for now..
    override fun requireSingleThreaded(): Boolean { return true }
}

fun main() {
    val stopWhen = 0.0001
    val pop = NEATPopulation(2, 1, 5)
    pop.reset()
    val alchemistScore = AlchemistScoreCalculation()
    val evolutionLoop = Stream.iterate(NEATUtil.constructNEATTrainer(pop, alchemistScore)) { train -> train.iteration(); println("here.."); train}
    val rightNetwork = evolutionLoop.takeWhile { it.error > stopWhen }
            .peek { println("Epoch #" + it.iteration + " Error:" + it.error + ", Species:" + pop.species.size) }
            .toList()
            .last()
    println(rightNetwork.bestGenome.score)
}
